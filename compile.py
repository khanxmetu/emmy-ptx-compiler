import json
import struct
import sys
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

from emmy.compiler.dtype import DataType
from emmy.compiler.dtype import get as get_dtype
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    Smem,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write


def load_ir_graph(ir_path: Path) -> tuple[Graph, dict]:
    with open(ir_path) as f:
        g = Graph.from_dict(json.load(f))

    meta_path = ir_path.parent / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return g, meta


PtxRegisterType = typing.Literal[
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
        self.reg_counts = {reg_type: 0 for reg_type in typing.get_args(PtxRegisterType)}
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
        *operands: Register | str | float,
        predicate: Predicate | None = None,
    ) -> None:
        self.instructions.append(PtxInstr(opcode, operands, predicate))


@dataclass
class PtxHeader:
    version: str = "8.0"
    target: str = "sm_89"
    address_size: typing.Literal[32, 64] = 64


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
    def __init__(self, graph: Graph, params: list[FuncParam], ir_kernel: KernelOp):
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

    def lower_index(
        self, indices: tuple[Expr, ...], shape: tuple[int, ...]
    ) -> Register:
        if not indices:
            index = self.regs.new("u64")
            self.add_instr("mov.u64", index, 0)
            return index
        index = self.coerce(self.lower_expr(indices[0]), "u64")
        # flatten into row major index
        for d, expr in enumerate(indices[1:], 1):
            coord = self.coerce(self.lower_expr(expr), "u64")
            flat = self.regs.new("u64")
            self.add_instr("mad.lo.u64", flat, index, shape[d], coord)
            index = flat
        return index

    def buffer_address(
        self, name: str, indices: tuple[Expr, ...], dtype: DataType | None = None
    ) -> tuple[Register, str, DataType]:
        # calculate byte address and resolve the buffer's memory space and dtype
        shared = self.ir_kernel.smem_buffers.get(name)
        if shared is not None:
            shape = shared.extents
            dtype = dtype or get_dtype(
                {"float": "f32", "__half": "f16", "double": "f64"}[shared.dtype]
            )
            base_address = self.regs.new("u64")
            self.add_instr("mov.u64", base_address, shared.name)
            memory_space = "shared"
        else:
            buffer = (
                self.graph.buffer(name) if dtype is None or len(indices) > 1 else None
            )
            dtype = dtype or buffer.dtype
            shape = (
                tuple(d.as_static() for d in buffer.shape) if len(indices) > 1 else ()
            )
            base_address = self.regs.get_reg(name)
            memory_space = "global"
        element_index = self.lower_index(indices, shape)
        address = self.regs.new("u64")
        self.add_instr("mad.lo.u64", address, element_index, dtype.nbytes, base_address)
        return address, memory_space, dtype

    @staticmethod
    def memory_types(dtype: DataType) -> tuple[PtxRegisterType, PtxRegisterType]:
        # tensor dtype -> PTX (memory, register) types
        # memory_type: load/store element type and byte width
        # register_type: register representation used for the value
        return {
            "f16": ("b16", "f16"),
            "f32": ("f32", "f32"),
            "f64": ("f64", "f64"),
            "i16": ("s16", "s32"),
            "i32": ("s32", "s32"),
            "i64": ("s64", "s64"),
            "u8": ("u8", "s32"),
            "u16": ("u16", "s32"),
            "u32": ("u32", "u32"),
            "u64": ("u64", "u64"),
            "bool": ("u8", "s32"),
        }[dtype.name]

    def lower_load(self, load: Load):
        address, memory_space, dtype = self.buffer_address(
            load.input, load.index, load.dtype
        )
        memory_type, register_type = self.memory_types(dtype)
        # TODO: group consecutive loads into vector instructions
        for element, name in enumerate(load.names):
            register = self.regs.new(register_type)
            byte_offset = element * dtype.nbytes
            self.add_instr(
                f"ld.{memory_space}.{memory_type}",
                register,
                f"[{address} + {byte_offset}]",
            )
            if dtype.name == "bool":
                register = self.as_pred(register)
            self.regs.bind(name, register)

    def lower_assign(self, assign: Assign): ...

    def lower_write(self, write: Write):
        address, memory_space, dtype = self.buffer_address(write.output, write.index)
        memory_type, register_type = self.memory_types(dtype)
        for element, name in enumerate(write.values):
            register = self.regs.get_reg(name)
            if dtype.name == "bool":
                register = self.as_pred(register)
            register = self.coerce(register, register_type)
            byte_offset = element * dtype.nbytes
            destination = f"[{address} + {byte_offset}]"
            if write.atomic:
                atomic_type = "u64" if memory_type == "s64" else memory_type
                self.add_instr(
                    f"red.{memory_space}.add.{atomic_type}", destination, register
                )
            else:
                self.add_instr(
                    f"st.{memory_space}.{memory_type}", destination, register
                )

    @staticmethod
    def numeric_type(dtype: PtxRegisterType) -> PtxRegisterType:
        # for smaller types use s32/f16 for computations
        if dtype in ("s16", "u8", "u16", "pred"):
            return "s32"
        if dtype in ("s32", "s64", "u32", "u64", "f32", "f64"):
            return dtype
        if dtype == "f16":
            return "f32"
        raise TypeError(f"unsupported scalar type: {dtype}")

    def coerce(self, reg: Register, dtype: PtxRegisterType) -> Register:
        # promote regtype to dtype
        if reg.type == dtype:
            return reg
        if dtype == "pred":
            return self.as_pred(reg)
        if reg.type == "pred":
            dst = self.regs.new("s32")
            self.add_instr("selp.s32", dst, 1, 0, reg)
            return self.coerce(dst, dtype)
        dst = self.regs.new(dtype)
        src_float, dst_float = reg.type.startswith("f"), dtype.startswith("f")
        rounding = ""
        if src_float and not dst_float:
            rounding = ".rzi"  # truncate toward zero
        elif dst_float:
            if not src_float:
                rounding = ".rn"  # integer to float: nearest-even
            elif int(dtype[1:]) < int(reg.type[1:]):
                rounding = ".rn"  # float narrowing: nearest-even
        self.add_instr(f"cvt{rounding}.{dtype}.{reg.type}", dst, reg)
        return dst

    def as_pred(self, reg: Register) -> Register:
        if reg.type == "pred":
            return reg
        reg = self.coerce(reg, self.numeric_type(reg.type))
        dst = self.regs.new("pred")
        # unordered comparison keeps NaN truthy
        cmp = "neu" if reg.type.startswith("f") else "ne"
        zero = (
            "0f00000000"
            if reg.type == "f32"
            else ("0d0000000000000000" if reg.type == "f64" else 0)
        )
        self.add_instr(f"setp.{cmp}.{reg.type}", dst, reg, zero)
        return dst

    def coerce_pair(self, a: Register, b: Register) -> tuple[Register, Register]:
        at, bt = self.numeric_type(a.type), self.numeric_type(b.type)
        if "f64" in (at, bt):
            dtype = "f64"
        elif "f32" in (at, bt):
            dtype = "f32"
        else:
            width = max(int(at[1:]), int(bt[1:]))
            unsigned = f"u{width}" in (at, bt)
            # according to c promotion rules when equal rank, convert both unsigned
            dtype = f"{'u' if unsigned else 's'}{width}"
        return self.coerce(a, dtype), self.coerce(b, dtype)

    def coerce_binary(
        self, op: str, lhs: Register, rhs: Register
    ) -> tuple[Register, Register]:
        if op in ("&&", "||"):
            return self.as_pred(lhs), self.as_pred(rhs)
        elif op in ("<<", ">>"):
            return (
                self.coerce(lhs, self.numeric_type(lhs.type)),
                # shift count type is independent of the value
                self.coerce(rhs, "u32"),
            )
        else:
            return self.coerce_pair(lhs, rhs)

    def lower_binary(self, op: str, lhs: Register, rhs: Register) -> Register:
        lhs, rhs = self.coerce_binary(op, lhs, rhs)
        comparisons = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq"}
        if op in comparisons:
            dst = self.regs.new("pred")
            self.add_instr(f"setp.{comparisons[op]}.{lhs.type}", dst, lhs, rhs)
            return dst
        floating = lhs.type.startswith("f")
        suffix = lhs.type
        match op:
            case "^":
                opcode = "xor"
                suffix = f"b{lhs.type[1:]}"
            case "&&":
                opcode = "and"
            case "||":
                opcode = "or"
            case "+":
                opcode = "add"
            case "-":
                opcode = "sub"
            case "*":
                opcode = "mul" if floating else "mul.lo"
            case "/":
                opcode = "div.rn" if floating else "div"
            case "//" if not floating:
                opcode = "div"
            case "%" if not floating:
                opcode = "rem"
            case _:
                raise NotImplementedError(f"binary lowering: {op}")
        dst = self.regs.new(lhs.type)
        self.add_instr(f"{opcode}.{suffix}", dst, lhs, rhs)
        return dst

    def lower_literal(self, value: int | float | bool, dtype: str) -> Register:  # noqa: PYI041
        def ptx_f32_literal(val):
            packed = struct.pack(">f", val)
            hex_digits = "".join(f"{b:02x}" for b in packed)
            return f"0f{hex_digits}"

        if isinstance(value, bool):
            dtype = "bool"
        match dtype:
            case "bool":
                reg = self.regs.new("pred")
                opcode = "setp.eq.s32" if value else "setp.ne.s32"
                self.add_instr(opcode, reg, 0, 0)
            case "float":
                reg = self.regs.new("f32")
                self.add_instr("mov.f32", reg, ptx_f32_literal(value))
            case "int":
                regtype = "s32" if -(1 << 31) <= value < (1 << 31) else "s64"
                reg = self.regs.new(regtype)
                self.add_instr(f"mov.{regtype}", reg, value)
        return reg

    def lower_expr(self, expr: Expr) -> Register:
        match expr:
            case Var(name=name):
                return self.regs.get_reg(name)

            case Literal(value=value, dtype=dtype):
                return self.lower_literal(value, dtype)

            case BinaryExpr(op=op, left=left, right=right):
                lhs = self.lower_expr(left)
                rhs = self.lower_expr(right)
                return self.lower_binary(op, lhs, rhs)

            case TernaryExpr(cond=cond, if_true=a, if_false=b):
                pred = self.as_pred(self.lower_expr(cond))
                lhs = self.lower_expr(a)
                rhs = self.lower_expr(b)
                if lhs.type == rhs.type == "pred":
                    dst = self.regs.new("pred")
                    self.add_instr("mov.pred", dst, rhs)
                    self.add_instr("mov.pred", dst, lhs, predicate=Predicate(pred))
                else:
                    lhs, rhs = self.coerce_pair(lhs, rhs)
                    dst = self.regs.new(lhs.type)
                    self.add_instr(f"selp.{lhs.type}", dst, lhs, rhs, pred)
                return dst
            case _:
                raise NotImplementedError(
                    f"unsupported expression: {type(expr).__name__}"
                )

    def lower_tile(self, tile: Tile):
        pprint(tile)
        gid = self.internal.gid
        N = tile.n_elements
        outofbounds = self.regs.new("pred")
        self.add_instr("setp.ge.u64", outofbounds, gid, N)
        self.add_instr("ret", predicate=Predicate(reg=outofbounds))

        def bind_axes():
            dim_prod = 1
            for i, axis in enumerate(reversed(tile.axes)):
                extent = axis.extent.as_static()
                axis_reg = gid
                if dim_prod > 1:
                    axis_reg = self.regs.new("u64")
                    self.add_instr("div.u64", axis_reg, gid, dim_prod)
                if i != len(tile.axes) - 1:
                    coord = self.regs.new("u64")
                    self.add_instr("rem.u64", coord, axis_reg, extent)
                    axis_reg = coord
                self.regs.bind(axis.name, axis_reg)
                dim_prod *= extent

        with self.regs.scope():
            bind_axes()
            self.lower_body(tile.body)

    def prologue(self) -> None:
        for param in self.params:
            addr_reg = self.regs.new("b64", param.name)
            self.add_instr("ld.param.u64", addr_reg, "[" + param.name + "]")
        for param in self.params:
            addr_reg = self.regs.get_reg(param.name)
            gaddr_reg = self.regs.new("b64", param.name)
            self.add_instr("cvta.to.global.u64", gaddr_reg, addr_reg)

        # kernel launches use only the x dimension
        # TODO: use u32 when gid < 2^32
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
            case Load():
                self.lower_load(stmt)
            case Write():
                self.lower_write(stmt)
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


if __name__ == "__main__":
    ir_path = Path(sys.argv[1])
    ir, meta = load_ir_graph(ir_path)
    codegen = PtxCodegen()
    ptx = codegen.generate(ir, meta)
    printer = PtxPrinter()
    print(printer.emit(ptx))
