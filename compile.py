import json
import math
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


@dataclass(frozen=True)
class PtxLabel:
    name: str

    def __str__(self) -> str:
        return self.name


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
    instructions: list[PtxInstr | PtxLabel] = field(default_factory=list)
    label_count: int = field(default=0, init=False)

    def new_label(self, name: str) -> PtxLabel:
        label = PtxLabel(f"%L_{name}_{self.label_count}")
        self.label_count += 1
        return label

    def add_label(self, label: PtxLabel) -> None:
        self.instructions.append(label)

    def add_instr(
        self,
        opcode: str,
        *operands: Register | PtxLabel | str | int | float,
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

    def lower_strided_loop(self, loop: StridedLoop):
        self.seed_accumulators(loop)
        if loop.end is None:
            end_expr = loop.axis.extent_expr()
        else:
            end_expr = loop.end
        # signed coordinates preserve negative differences in the body
        start_src = self.lower_coordinate(loop.start)
        step = self.lower_coordinate(loop.step)
        end = self.lower_coordinate(end_expr)

        start = self.regs.new("s64")
        self.add_instr("mov.s64", start, start_src)

        head = self.func.new_label("loop")
        done = self.func.new_label("loop_end")

        self.func.add_label(head)
        finished = self.lower_binary(">=", start, end)
        self.add_instr("bra", done, predicate=Predicate(finished))
        with self.regs.scope():
            self.regs.bind(loop.axis.name, start)
            self.lower_body(loop.body)
        self.add_instr("add.s64", start, start, step)
        self.add_instr("bra", head)
        self.func.add_label(done)

    def seed_accumulators(self, loop: StridedLoop):
        if not loop.seed:
            return
        seen = set()
        for stmt in loop.body:
            if not isinstance(stmt, Accum) or stmt.name in seen:
                continue
            seen.add(stmt.name)
            identity = stmt.op.identity
            _, reg_type = self.memory_types(stmt.dtype)
            reg = self.lower_constant(identity, reg_type)
            self.regs.bind(stmt.name, reg)

    def lower_accum(self, accum: Accum):
        dst = self.regs.get_reg(accum.name)
        if accum.base is None:
            base = dst
        else:
            base = self.regs.get_reg(accum.base)
        value = self.regs.get_reg(accum.value)
        src = self.lower_elementwise(accum.op.reduce_canon, [base, value], dst.type)
        src = self.coerce(src, dst.type)

        # update the loop-carried register without rebinding its name
        if dst.type == "f16":
            self.add_instr("mov.b16", dst, src)
        else:
            self.add_instr(f"mov.{dst.type}", dst, src)

    def lower_warp_shuffle(self, warp: WarpShuffle): ...

    def lower_smem(self, smem: Smem): ...

    def lower_cond(self, cond: Cond): ...

    def lower_sync(self, sync: Sync): ...

    def lower_tree_halve(self, tree_halve: TreeHalve): ...

    def lower_index(
        self, indices: tuple[Expr, ...], shape: tuple[int, ...]
    ) -> Register:
        if not indices:
            return self.lower_constant(0, "u64")
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
            base = self.regs.new("u64")
            self.add_instr("mov.u64", base, shared.name)
            space = "shared"
        else:
            buffer = (
                self.graph.buffer(name) if dtype is None or len(indices) > 1 else None
            )
            dtype = dtype or buffer.dtype
            shape = (
                tuple(d.as_static() for d in buffer.shape) if len(indices) > 1 else ()
            )
            base = self.regs.get_reg(name)
            space = "global"
        index = self.lower_index(indices, shape)
        addr = self.regs.new("u64")
        self.add_instr("mad.lo.u64", addr, index, dtype.nbytes, base)
        return addr, space, dtype

    @staticmethod
    def memory_types(dtype: DataType) -> tuple[PtxRegisterType, PtxRegisterType]:
        # tensor dtype -> PTX (memory, register) types
        # mem_type: load/store element type and byte width
        # reg_type: register representation used for the value
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
        addr, space, dtype = self.buffer_address(load.input, load.index, load.dtype)
        mem_type, reg_type = self.memory_types(dtype)
        # TODO: group consecutive loads into vector instructions
        for i, name in enumerate(load.names):
            reg = self.regs.new(reg_type)
            offset = i * dtype.nbytes
            self.add_instr(
                f"ld.{space}.{mem_type}",
                reg,
                f"[{addr} + {offset}]",
            )
            if dtype.name == "bool":
                reg = self.as_pred(reg)
            self.regs.bind(name, reg)

    def lower_assign(self, assign: Assign):
        
        # map emmy names to ptx types; unsigned and float names already match
        dst_type = {
            "i16": "s16",
            "i32": "s32",
            "i64": "s64",
            "bool": "pred",
        }.get(assign.dtype.name, assign.dtype.name)
        args = [self.regs.get_reg(name) for name in assign.args]
        if assign.op.name == "copy":
            dst = args[0]
        else:
            dst = self.lower_elementwise(assign.op.name, args, dst_type)
        self.regs.bind(assign.name, self.coerce(dst, dst_type))

    def lower_elementwise(
        self, op: str, args: list[Register], dst_type: PtxRegisterType
    ) -> Register:
        if op in ("add", "subtract", "multiply", "maximum", "minimum"):
            compute_type = self.numeric_type(dst_type)
        else:
            # simplify floating-point computation to f32
            # TODO: preserve wider computation precision
            compute_type = "f32"

        args = [self.coerce(reg, compute_type) for reg in args]
        dst = self.regs.new(compute_type)
        match op:
            case "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum":
                opcode = {
                    "add": "add",
                    "subtract": "sub",
                    "multiply": "mul",
                    "divide": "div.rn",
                    "maximum": "max",
                    "minimum": "min",
                }[op]
                if op == "multiply" and not compute_type.startswith("f"):
                    opcode = "mul.lo"
                self.add_instr(f"{opcode}.{compute_type}", dst, *args)
            case "exp":
                scaled = self.regs.new("f32")
                # exp(x) = 2 ** (x * log2(e))
                self.add_instr("mul.f32", scaled, args[0], math.log2(math.e))
                self.add_instr("ex2.approx.f32", dst, scaled)
            case "reciprocal":
                self.add_instr("rcp.rn.f32", dst, args[0])
            case "relu":
                self.add_instr("max.f32", dst, 0.0, args[0])
            case "sqrt":
                self.add_instr("sqrt.rn.f32", dst, args[0])
            case _:
                raise NotImplementedError(f"assign lowering: {op}")
        return dst

    def lower_write(self, write: Write):
        addr, space, dtype = self.buffer_address(write.output, write.index)
        mem_type, reg_type = self.memory_types(dtype)
        # TODO grouped write should become vector store
        for i, name in enumerate(write.values):
            reg = self.regs.get_reg(name)
            if dtype.name == "bool":
                reg = self.as_pred(reg)
            reg = self.coerce(reg, reg_type)
            offset = i * dtype.nbytes
            dst = f"[{addr} + {offset}]"
            if write.atomic:
                atomic_type = "u64" if mem_type == "s64" else mem_type
                self.add_instr(f"red.{space}.add.{atomic_type}", dst, reg)
            else:
                self.add_instr(f"st.{space}.{mem_type}", dst, reg)

    @staticmethod
    def numeric_type(dtype: PtxRegisterType) -> PtxRegisterType:
        # c style integer/float promotion
        # for smaller types use s32/f32 for computations
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
        dst = self.regs.new("s32" if dtype == "u8" else dtype)
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
        zero = 0.0 if reg.type.startswith("f") else 0
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

    def lower_constant(self, value: int | float | bool, dtype: PtxRegisterType) -> Register:
        # materialize a constant in the requested ptx type
        if dtype == "pred":
            src_type = "s32"
            value = bool(value)
        elif dtype == "f16":
            src_type = "f32"
        else:
            src_type = dtype

        if src_type.startswith("f"):
            value = float(value)
        else:
            value = int(value)

        src = self.regs.new(src_type)
        self.add_instr(f"mov.{src_type}", src, value)
        return self.coerce(src, dtype)

    def lower_literal(self, value: int | float | bool, dtype: str) -> Register:  # noqa: PYI041
        # choose a ptx type for an ir literal
        if isinstance(value, bool) or dtype == "bool":
            reg_type = "pred"
        elif dtype == "float":
            reg_type = "f32"
        elif -(1 << 31) <= value < (1 << 31):
            reg_type = "s32"
        else:
            reg_type = "s64"
        return self.lower_constant(value, reg_type)

    def lower_coordinate(self, expr: Expr) -> Register:
        # keep coordinate arithmetic signed and 64-bit
        if isinstance(expr, Literal):
            return self.lower_constant(expr.value, "s64")
        return self.coerce(self.lower_expr(expr), "s64")

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
        # pprint(tile)
        gid = self.internal.gid
        N = tile.n_elements
        outofbounds = self.regs.new("pred")
        self.add_instr("setp.ge.u64", outofbounds, gid, N)
        self.add_instr("ret", predicate=Predicate(reg=outofbounds))

        def bind_axes():
            if not tile.axes:
                return
            # decode once into signed coordinate arithmetic
            coord_gid = self.coerce(gid, "s64")
            dim_prod = 1
            for i, axis in enumerate(reversed(tile.axes)):
                extent = axis.extent.as_static()
                if dim_prod > 1:
                    axis_reg = self.regs.new("s64")
                    self.add_instr("div.s64", axis_reg, coord_gid, dim_prod)
                else:
                    axis_reg = coord_gid
                if i != len(tile.axes) - 1:
                    coord = self.regs.new("s64")
                    self.add_instr("rem.s64", coord, axis_reg, extent)
                    axis_reg = coord
                self.regs.bind(axis.name, axis_reg)
                dim_prod *= extent

        with self.regs.scope():
            bind_axes()
            self.lower_body(tile.body)

    def prologue(self) -> None:
        for param in self.params:
            gaddr_reg = self.regs.new("b64", param.name)
            addr_reg = self.regs.new("b64")
            self.add_instr("ld.param.u64", addr_reg, "[" + param.name + "]")
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
            case Assign():
                self.lower_assign(stmt)
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

    def instruction(self, instr: PtxInstr | PtxLabel) -> str:
        if isinstance(instr, PtxLabel):
            return f"{instr.name}:"

        prefix = ""
        if instr.predicate is not None:
            pred = instr.predicate
            prefix = f"@{'!' if pred.negated else ''}{pred.reg} "

        operands = ", ".join(
            str(int(op)) if isinstance(op, bool) else str(op) for op in instr.operands
        )
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
