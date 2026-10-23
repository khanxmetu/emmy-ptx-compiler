import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.kernel.ir import KernelOp


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


class RegisterStore:
    def __init__(self):
        self.reg_counts = {reg_type: 0 for reg_type in get_args(PtxRegisterType)}
        self.bindings: dict[str, str] = {}

    @staticmethod
    def get_prefix(reg_type: PtxRegisterType) -> str:
        return f"{reg_type}_"

    def bind(self, varname: str, regname: str):
        self.bindings[varname] = regname

    def get_reg(self, varname) -> str:
        return self.bindings[varname]

    def new(self, reg_type: PtxRegisterType, varname: str | None = None) -> str:
        count = self.reg_counts[reg_type]
        regname = f"%{self.get_prefix(reg_type)}{count}"
        self.reg_counts[reg_type] += 1
        if varname is not None:
            self.bind(varname, regname)
        return regname


@dataclass
class Predicate:
    reg: str
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
        *operands: str,
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


class PtxCodegen:
    def __init__(self):
        pass

    def get_func_params(
        self, node_id: str, codegen_ctx: PtxCodegenCtx
    ) -> list[FuncParam]:
        param_names = codegen_ctx.argnames[node_id]
        params = [
            FuncParam(name, type="u64", pointee_type=codegen_ctx.ir.buffer(name).dtype)
            for name in param_names
        ]
        return params

    def kernel_prologue(
        self, node_id: str, codegen_ctx: PtxCodegenCtx, ptx_kernel: PtxFunction
    ) -> None:
        params = self.get_func_params(node_id, codegen_ctx)
        ptx_kernel.params = params
        regs = ptx_kernel.registers
        #  load and convert params addresses
        for param in params:
            addr_reg = regs.new("b64", param.name)
            ptx_kernel.add_instr("ld.param.u64", addr_reg, "[" + param.name + "]")
        for param in params:
            addr_reg = regs.get_reg(param.name)
            gaddr_reg = regs.new("b64", param.name)
            ptx_kernel.add_instr("cvta.to.global.u64", gaddr_reg, addr_reg)

        # calculate gid
        # TODO: optimize for when gid < 2^32
        tid = regs.new("u32")
        bid = regs.new("u32")
        bdim = regs.new("u32")
        tid64 = regs.new("u64")
        gid = regs.new("u64", "gid")
        warp = regs.new("u32", "warp")
        lane = regs.new("u32", "lane")

        ptx_kernel.add_instr("mov.u32", tid, "%tid.x")
        ptx_kernel.add_instr("mov.u32", bid, "%ctaid.x")
        ptx_kernel.add_instr("mov.u32", bdim, "%ntid.x")
        ptx_kernel.add_instr("cvt.u64.u32", tid64, tid)
        ptx_kernel.add_instr("mad.wide.u32", gid, bdim, bid, tid64)

        ptx_kernel.add_instr("shr.u32", warp, tid, 5)
        ptx_kernel.add_instr("and.b32", lane, tid, 31)

    def translate_kernel(
        self, node_id: str, kernel: KernelOp, codegen_ctx: PtxCodegenCtx
    ) -> PtxFunction:
        ptx_kernel = PtxFunction(name=kernel.name)
        self.kernel_prologue(node_id, codegen_ctx, ptx_kernel)

        ptx_kernel.add_instr("ret")
        return ptx_kernel

    def generate(self, ir: Graph, metadata: dict) -> PtxModule:
        argnames = {
            kernel["node"]: kernel["arg_order"] for kernel in metadata["kernels"]
        }
        ctx = PtxCodegenCtx(argnames, ir)
        module = PtxModule()
        for node in ir.nodes.values():
            if isinstance(node.op, KernelOp):
                kernel = self.translate_kernel(node.id, node.op, ctx)
                module.kernels.append(kernel)
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
        return f"{instr.opcode:25} {', '.join([str(op) for op in instr.operands])};"

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
