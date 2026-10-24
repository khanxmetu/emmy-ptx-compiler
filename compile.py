import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Literal, get_args

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    Smem,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum


def load_ir_graph(ir_path: Path) -> tuple[Graph, dict]:
    with open(ir_path) as f:
        g = Graph.from_dict(json.load(f))

    meta_path = ir_path.parent / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return g, meta


PtxRegisterType = Literal[
    "s8",
    "s16",
    "s32",
    "s64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f16",
    "f16x2",
    "f32",
    "f64",
    "b8",
    "b16",
    "b32",
    "b64",
    "b128",
    "pred",
]


@dataclass(frozen=True)
class Register:
    name: str
    type: PtxRegisterType

    def __str__(self) -> str:
        return self.name


class RegisterStore:
    def __init__(self):
        self.reg_counts = {reg_type: 0 for reg_type in get_args(PtxRegisterType)}
        self.bindings: dict[str, Register] = {}

    @staticmethod
    def get_prefix(reg_type: PtxRegisterType) -> str:
        return f"{reg_type}_"

    def bind(self, name: str, register: Register):
        self.bindings[name] = register

    def get_reg(self, name: str) -> Register:
        return self.bindings[name]

    def new(self, reg_type: PtxRegisterType, varname: str | None = None) -> Register:
        count = self.reg_counts[reg_type]
        register = Register(f"%{self.get_prefix(reg_type)}{count}", reg_type)
        self.reg_counts[reg_type] += 1
        if varname is not None:
            self.bind(varname, register)
        return register

    @contextmanager
    def scope(self):
        outer = self.bindings
        self.bindings = outer.copy()
        try:
            yield
        finally:
            self.bindings = outer


@dataclass
class Predicate:
    reg: Register
    negated: bool = False


@dataclass
class PtxInstr:
    opcode: str
    operands: tuple = field(default_factory=tuple)
    predicate: Predicate | None = None


@dataclass
class FuncParam:
    name: str
    type: PtxRegisterType
    pointee_type: PtxRegisterType | None = None


@dataclass
class PtxFunction:
    name: str
    params: list[FuncParam] = field(default_factory=list)
    registers: RegisterStore = field(default_factory=RegisterStore)
    instructions: list[PtxInstr] = field(default_factory=list)

    def add_instr(
        self,
        opcode: str,
        *operands: Register | str | int | float,
        predicate: Predicate | None = None,
    ) -> None:
        self.instructions.append(PtxInstr(opcode, operands, predicate))


@dataclass
class PtxHeader:
    version: str = "8.0"
    target: str = "sm_89"
    address_size: Literal[32, 64] = 64


@dataclass
class PtxModule:
    header: PtxHeader = field(default_factory=PtxHeader)
    kernels: list[PtxFunction] = field(default_factory=list)


@dataclass
class PtxCodegenCtx:
    argnames: dict[str, list[str]]
    ir: Graph


@dataclass
class KernelRegisters:
    gid: Register
    warp: Register
    lane: Register


class KernelLowerer:
    def __init__(self, graph, params: list[FuncParam], ir_kernel: KernelOp):
        self.graph = graph
        self.ir_kernel = ir_kernel
        self.func = PtxFunction(ir_kernel.name, params)

        self.regs = self.func.registers
        self.params = self.func.params
        self.add_instr = self.func.add_instr
        self.internal: KernelRegisters

    def lower_strided_loop(self, loop: StridedLoop): ...

    def lower_accum(self, accum: Accum): ...

    def lower_warp_shuffle(self, warp: WarpShuffle): ...

    def lower_smem(self, smem: Smem): ...

    def lower_cond(self, cond: Cond): ...

    def lower_sync(self, sync: Sync): ...

    def lower_tree_halve(self, tree_halve: TreeHalve): ...

    def lower_tile(self, tile: Tile):
        pprint(tile)
        gid = self.internal.gid
        N = tile.n_elements
        outofbounds = self.regs.new("pred")
        self.add_instr("setp.ge.u64", outofbounds, gid, N)
        self.add_instr("ret", predicate=Predicate(reg=outofbounds))

        def divmod_64(a, b) -> tuple[Register, Register]:
            """Returns regs holding u64 quotient and remainder."""
            q = self.regs.new("u64")
            r = self.regs.new("u64")
            prod = self.regs.new("u64")

            self.add_instr("div.u64", q, a, b)
            self.add_instr("mul.lo.u64", prod, q, b)
            self.add_instr("sub.u64", r, a, prod)
            return q, r

        def bind_axes():
            # Decode per axis indices from gid, innermost axis is fastest moving
            # e.g axis: (i, j, k) => gid = k + Dk*(j + Dj*i)
            # TODO: optimize: use u32 if gid<2^32. try optimized instructions when axis.extent is power of two.
            remaining = self.regs.new("u64")
            self.add_instr("mov.b64", remaining, gid)

            for axis in reversed(tile.axes[1:]):
                remaining, axis_reg = divmod_64(remaining, axis.extent.as_static())
                self.regs.bind(axis.name, axis_reg)
            if tile.axes:
                self.regs.bind(tile.axes[0].name, remaining)

        with self.regs.scope():
            bind_axes()
            self.lower_body(tile.body)

    def prologue(self) -> None:
        #  load and convert params addresses
        for param in self.params:
            addr_reg = self.regs.new("b64", param.name)
            self.add_instr("ld.param.u64", addr_reg, "[" + param.name + "]")
        for param in self.params:
            addr_reg = self.regs.get_reg(param.name)
            gaddr_reg = self.regs.new("b64", param.name)
            self.add_instr("cvta.to.global.u64", gaddr_reg, addr_reg)

        # calculate gid, kernel ir always launches with blockDim(x, 1, 1) and gridDim(x, 1, 1)
        # TODO: optimize for when gid < 2^32
        tid = self.regs.new("u32")
        bid = self.regs.new("u32")
        bdim = self.regs.new("u32")
        tid64 = self.regs.new("u64")
        gid = self.regs.new("u64")
        warp = self.regs.new("u32")
        lane = self.regs.new("u32")
        self.internal = KernelRegisters(gid=gid, warp=warp, lane=lane)

        self.add_instr("mov.u32", tid, "%tid.x")
        self.add_instr("mov.u32", bid, "%ctaid.x")
        self.add_instr("mov.u32", bdim, "%ntid.x")
        self.add_instr("cvt.u64.u32", tid64, tid)
        self.add_instr("mad.wide.u32", gid, bdim, bid, tid64)

        self.add_instr("shr.u32", warp, tid, 5)
        self.add_instr("and.b32", lane, tid, 31)

    def lower_body(self, body) -> None:
        for child in body:
            self.lower_stmt(child)

    def lower_stmt(self, stmt) -> None:
        match stmt:
            case Tile():
                self.lower_tile(stmt)
            case StridedLoop():
                self.lower_strided_loop(stmt)
            case Accum():
                self.lower_accum(stmt)
            case WarpShuffle():
                self.lower_warp_shuffle(stmt)
            case Smem():
                self.lower_smem(stmt)
            case Cond():
                self.lower_cond(stmt)
            case Sync():
                self.lower_sync(stmt)
            case TreeHalve():
                self.lower_tree_halve(stmt)
            case _:
                raise NotImplementedError(
                    f"{self.ir_kernel.name}: unsupported {type(stmt).__name__}"
                )

    def lower(self) -> PtxFunction:
        self.prologue()
        self.lower_body(self.ir_kernel.body)
        self.add_instr("ret")
        return self.func


class PtxCodegen:
    def __init__(self):
        pass

    def get_func_params(self, argnames: list[str], ir: Graph) -> list[FuncParam]:
        params = [
            FuncParam(name, type="u64", pointee_type=ir.buffer(name).dtype)
            for name in argnames
        ]
        return params

    def generate(self, ir: Graph, metadata: dict) -> PtxModule:
        argnames = {
            kernel["node"]: kernel["arg_order"] for kernel in metadata["kernels"]
        }
        module = PtxModule()
        for node in ir.nodes.values():
            if isinstance(node.op, KernelOp):
                params = self.get_func_params(argnames[node.id], ir)
                lowerer = KernelLowerer(ir, params, node.op)
                ptx_kernel = lowerer.lower()
                module.kernels.append(ptx_kernel)
        return module


class PtxPrinter:
    def header(self, header: PtxHeader) -> list[str]:
        return [f".{k} {v}" for k, v in header.__dict__.items()]

    def signature(self, kernel: PtxFunction) -> list[str]:
        return [
            f".visible .entry {kernel.name}(",
            ",\n".join(
                [f"    .param .{param.type} {param.name}" for param in kernel.params]
            ),
            ")",
        ]

    def registers(self, registers: RegisterStore) -> list[str]:
        return [
            f".reg .{reg_type:19} %{RegisterStore.get_prefix(reg_type)}<{count}>;"
            for reg_type, count in registers.reg_counts.items()
            if count
        ]

    def instruction(self, instr: PtxInstr) -> list[str]:
        prefix = ""
        if instr.predicate is not None:
            pred = instr.predicate
            prefix = f"@{'!' if pred.negated else ''}{pred.reg} "

        operands = ", ".join(str(op) for op in instr.operands)
        return f"{prefix}{instr.opcode} {operands};"

    def kernel(self, kernel: PtxFunction) -> list[str]:
        return [
            *self.signature(kernel),
            "{",
            *self.registers(kernel.registers),
            *(self.instruction(i) for i in kernel.instructions),
            "}",
        ]

    def module(self, module: PtxModule) -> list[str]:
        lines = self.header(module.header)
        for kernel in module.kernels:
            lines.extend(self.kernel(kernel))
        return lines

    def emit(self, module: PtxModule) -> str:
        return "\n".join(self.module(module))


ir_path = Path(sys.argv[1])
ir, meta = load_ir_graph(ir_path)
codegen = PtxCodegen()
ptx = codegen.generate(ir, meta)
printer = PtxPrinter()
print(printer.emit(ptx))
