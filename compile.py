"""Kernel IR -> PTX compiler.

Usage: python compile.py path/to/kernel.ir.json > out.ptx
"""
import json
import struct
import os
import sys

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write

# static .shared is capped at 48 KiB; above that the launch supplies a dynamic pool
STATIC_SMEM_LIMIT = 48 * 1024
SMEM_POOL = "_smem_pool"

# PTX register kinds: prefix -> declared type
REG_TYPES = {"p": ".pred", "r": ".b32", "rd": ".b64", "f": ".f32"}

# Instruction selection for Assign/Accum ops: (op name, dtype) -> PTX templates.
# {d} is the destination register, {a}/{b} the source registers. Most ops are one
# instruction; exp needs two because PTX only has a base-2 exponential (legalization).
ELEMENTWISE = {
    ("add", "f32"):        ["add.f32 {d}, {a}, {b};"],
    ("subtract", "f32"):   ["sub.f32 {d}, {a}, {b};"],
    ("multiply", "f32"):   ["mul.f32 {d}, {a}, {b};"],
    ("divide", "f32"):     ["div.rn.f32 {d}, {a}, {b};"],
    ("maximum", "f32"):    ["max.f32 {d}, {a}, {b};"],
    ("minimum", "f32"):    ["min.f32 {d}, {a}, {b};"],
    ("relu", "f32"):       ["max.f32 {d}, {a}, 0f00000000;"],
    ("copy", "f32"):       ["mov.f32 {d}, {a};"],
    ("negative", "f32"):   ["neg.f32 {d}, {a};"],
    ("abs", "f32"):        ["abs.f32 {d}, {a};"],
    ("square", "f32"):     ["mul.f32 {d}, {a}, {a};"],
    ("reciprocal", "f32"): ["rcp.rn.f32 {d}, {a};"],
    ("sqrt", "f32"):       ["sqrt.rn.f32 {d}, {a};"],
    ("rsqrt", "f32"):      ["rsqrt.approx.f32 {d}, {a};"],
    ("exp", "f32"):        ["mul.f32 {d}, {a}, 0f3FB8AA3B;",  # x * log2(e)
                            "ex2.approx.f32 {d}, {d};"],
    ("exp_fast", "f32"):   ["mul.f32 {d}, {a}, 0f3FB8AA3B;",  # emmy's __expf: same lowering
                            "ex2.approx.f32 {d}, {d};"],
    ("log", "f32"):        ["lg2.approx.f32 {d}, {a};",       # log2(x) * ln(2)
                            "mul.f32 {d}, {d}, 0f3F317218;"],
    ("tanh", "f32"):       ["tanh.approx.f32 {d}, {a};"],
    ("sigmoid", "f32"):    ["mul.f32 {d}, {a}, 0fBFB8AA3B;",  # 1 / (1 + exp(-x)): x * -log2(e)
                            "ex2.approx.f32 {d}, {d};",
                            "add.f32 {d}, {d}, 0f3F800000;",
                            "rcp.rn.f32 {d}, {d};"],
}

# Instruction selection for BinaryExpr in index expressions (32-bit ints):
# op -> (result register kind, PTX template). Comparisons produce predicates.
INDEX_OPS = {
    "+":  ("r", "add.s32 {d}, {a}, {b};"),
    "-":  ("r", "sub.s32 {d}, {a}, {b};"),
    "*":  ("r", "mul.lo.s32 {d}, {a}, {b};"),
    # indices are >= 0, so unsigned div/rem are exact and ptxas turns a power-of-two
    # divisor into one shift / and (signed needs a sign-fixup sequence)
    "/":  ("r", "div.u32 {d}, {a}, {b};"),
    "//": ("r", "div.u32 {d}, {a}, {b};"),    # emmy's row-major flatten
    "%":  ("r", "rem.u32 {d}, {a}, {b};"),
    "<":  ("p", "setp.lt.s32 {d}, {a}, {b};"),
    "<=": ("p", "setp.le.s32 {d}, {a}, {b};"),
    ">":  ("p", "setp.gt.s32 {d}, {a}, {b};"),
    ">=": ("p", "setp.ge.s32 {d}, {a}, {b};"),
    "==": ("p", "setp.eq.s32 {d}, {a}, {b};"),
    "!=": ("p", "setp.ne.s32 {d}, {a}, {b};"),
    "&&": ("p", "and.pred {d}, {a}, {b};"),
    "||": ("p", "or.pred {d}, {a}, {b};"),
}

# f32 comparisons produce a predicate (`bool v = greater(a, b)` -> setp.gt.f32 %p, a, b)
COMPARE = {"greater": "gt", "less": "lt", "greater_equal": "ge", "less_equal": "le",
           "equal": "eq", "not_equal": "ne"}


def select_elementwise(op, dtype, d, srcs):
    """Instantiate the ELEMENTWISE templates for one op with concrete registers."""
    try:
        templates = ELEMENTWISE[(op, dtype)]
    except KeyError:
        raise NotImplementedError(f"no PTX selection for {op} on {dtype}")
    a, b = (list(srcs) + [None])[:2]
    return [t.format(d=d, a=a, b=b) for t in templates]


class PtxKernel:
    """Collects one kernel's PTX in sections; to_text() assembles them at the end."""

    def __init__(self, entry, params):
        self.entry = entry      # entry name = KernelOp.name
        self.params = params    # buffer names in meta.json arg_order
        self.shared = []        # .shared declarations (filled by Smem later)
        self.reg_counts = {}    # register kind -> number handed out so far
        self.body = []          # instruction lines, in emission order
        self.buffers = {}       # buffer name -> register holding its global address
        self.shapes = {}        # global buffer name -> shape (from meta.json args)
        self.dtypes = {}        # global buffer name -> dtype string (from meta.json args)
        self.smem = {}          # shared array name -> extents; its PTX symbol is the same name
        self.smem_offset = {}   # shared array name -> byte offset in the dynamic pool (pool mode)
        self.env = {}           # SSA / axis name -> register holding its value
        self.labels = 0         # counter for unique branch labels

    def __repr__(self):
        return f"PtxKernel({self.entry}, buffers={self.buffers}, env={self.env})"

    def fresh(self, kind):
        """Return a new virtual register of the given kind, e.g. fresh("f") -> "%f3"."""
        n = self.reg_counts.get(kind, 0) + 1
        self.reg_counts[kind] = n
        return f"%{kind}{n}"

    def fresh_label(self, prefix):
        """Return a unique label name, e.g. fresh_label("LOOP") -> "LOOP_2"."""
        self.labels += 1
        return f"{prefix}_{self.labels}"

    def emit(self, line):
        """Append one instruction (or label/comment) to the body."""
        self.body.append(line)

    def to_text(self):
        params = ",\n".join(f"    .param .u64 param_{name}" for name in self.params)
        # registers start at 1, so %f<N+1> declares %f0..%fN
        regs = [f"    .reg {REG_TYPES[kind]} %{kind}<{n + 1}>;"
                for kind, n in self.reg_counts.items()]
        body = [line if line.endswith(":") else f"    {line}" for line in self.body]
        return "\n".join([
            ".version 8.0",
            ".target sm_89",
            ".address_size 64",
            "",
            *self.shared,
            "",
            f".visible .entry {self.entry}(",
            params,
            ")",
            "{",
            *regs,
            "",
            *body,
            "}",
            "",
        ])


def load_kernel(ir_path):
    """Deserialize the IR graph and return its single KernelOp."""
    graph = Graph.from_dict(json.load(open(ir_path)))
    for node in graph.nodes.values():
        if type(node.op).__name__ == "KernelOp":
            return node.op
    sys.exit(f"no KernelOp in {ir_path}")


def load_arg_order(ir_path, entry):
    """meta.json sits next to the IR; its arg_order is the parameter contract."""
    meta = json.load(open(os.path.join(os.path.dirname(ir_path), "meta.json")))
    for kernel in meta["kernels"]:
        if kernel["entry"] == entry:
            return kernel["arg_order"]
    sys.exit(f"no kernel {entry} in meta.json")


def load_buffer_shapes(ir_path, entry):
    """Buffer name -> (shape, dtype), from meta.json's args.

    The shape flattens multi-dim indices. The dtype is the buffer's real element type: the
    IR's Write.value_dtype says f32 even when the output buffer is f16 (emmy converts on store),
    so meta.json is the only place a non-f32 buffer can be detected and refused.
    """
    meta = json.load(open(os.path.join(os.path.dirname(ir_path), "meta.json")))
    for kernel in meta["kernels"]:
        if kernel["entry"] == entry:
            return ({arg["name"]: arg["shape"] for arg in kernel["args"]},
                    {arg["name"]: arg["dtype"] for arg in kernel["args"]})
    sys.exit(f"no kernel {entry} in meta.json")


def emit_prologue(k):
    """Load each pointer param and convert it to a global-space address."""
    for name in k.params:
        param = k.fresh("rd")
        k.emit(f"ld.param.u64 {param}, [param_{name}];")
        addr = k.fresh("rd")
        k.emit(f"cvta.to.global.u64 {addr}, {param};")
        k.buffers[name] = addr


# ---------------------------------------------------------------- expressions

def float_literal(value):
    """PTX float immediate as exact IEEE hex, e.g. 1.0 -> 0f3F800000 (also handles inf)."""
    return "0f" + struct.pack(">f", value).hex().upper()


def in_register(k, operand):
    """PTX allows immediates only in some operand slots; move an int immediate into a %r."""
    if operand.startswith("%"):
        return operand
    reg = k.fresh("r")
    k.emit(f"mov.s32 {reg}, {operand};")
    return reg


def as_pred(k, operand):
    """Comparisons already yield a %p; any other int value is tested against 0."""
    if operand.startswith("%p"):
        return operand
    pred = k.fresh("p")
    k.emit(f"setp.ne.s32 {pred}, {in_register(k, operand)}, 0;")
    return pred


def emit_expr(k, e):
    """Lower an index expression (post-order) and return the register or immediate holding it."""
    kind = type(e).__name__
    if kind == "Var":
        return k.env[e.name]
    if kind == "Literal":
        return float_literal(e.value) if e.dtype == "float" else str(e.value)
    if kind == "BinaryExpr":
        if e.op in ("&&", "||"):                     # and.pred/or.pred take predicates only
            left, right = as_pred(k, emit_expr(k, e.left)), as_pred(k, emit_expr(k, e.right))
        else:
            left = in_register(k, emit_expr(k, e.left))  # first operand must be a register
            right = emit_expr(k, e.right)
        reg_kind, template = INDEX_OPS[e.op]
        d = k.fresh(reg_kind)
        k.emit(template.format(d=d, a=left, b=right))
        return d
    if kind == "TernaryExpr":
        cond = emit_expr(k, e.cond)
        if_true, if_false = emit_expr(k, e.if_true), emit_expr(k, e.if_false)
        cond = as_pred(k, cond)
        if if_true.startswith(("%f", "0f")):   # a float ternary (Let-bound): select floats
            d = k.fresh("f")
            k.emit(f"selp.f32 {d}, {if_true}, {if_false}, {cond};")
            return d
        d = k.fresh("r")
        k.emit(f"selp.s32 {d}, {if_true}, {if_false}, {cond};")
        return d
    raise NotImplementedError(f"no lowering for expression {kind}")


def flat_index(k, index, shape):
    """Row-major flatten of a multi-dim index against the buffer's trailing dims.

    x[a0, a1] with shape (64, 512) -> a0 * 512 + a1. An empty index means element 0.
    """
    if not index:
        return "0"
    dims = shape[len(shape) - len(index):]
    flat = emit_expr(k, index[0])
    for idx, extent in zip(index[1:], dims[1:]):
        scaled = k.fresh("r")
        k.emit(f"mul.lo.s32 {scaled}, {in_register(k, flat)}, {extent};")
        flat = k.fresh("r")
        k.emit(f"add.s32 {flat}, {scaled}, {emit_expr(k, idx)};")
    return flat


def emit_address(k, buffer, index, elem_bytes):
    """Return (state space, address register) for buffer[index].

    Global buffers use 64-bit addresses: base + idx * bytes (mul.wide widens the 32-bit index).
    Shared arrays use 32-bit addresses relative to their .shared symbol.
    """
    if buffer in k.smem:
        flat = flat_index(k, index, k.smem[buffer])
        base = k.fresh("r")
        if buffer in k.smem_offset:          # pool mode: symbol is the pool, array is an offset
            k.emit(f"mov.u32 {base}, {SMEM_POOL};")
            if k.smem_offset[buffer]:
                k.emit(f"add.s32 {base}, {base}, {k.smem_offset[buffer]};")
        else:
            k.emit(f"mov.u32 {base}, {buffer};")
        if flat == "0":
            return "shared", base
        addr = k.fresh("r")
        k.emit(f"mad.lo.s32 {addr}, {in_register(k, flat)}, {elem_bytes}, {base};")
        return "shared", addr

    if k.dtypes.get(buffer, "f32") != "f32":
        raise NotImplementedError(f"buffer {buffer} is {k.dtypes[buffer]}; only f32 buffers are lowered")
    flat = flat_index(k, index, k.shapes[buffer])
    base = k.buffers[buffer]
    if flat == "0":
        return "global", base
    offset, addr = k.fresh("rd"), k.fresh("rd")
    k.emit(f"mul.wide.s32 {offset}, {in_register(k, flat)}, {elem_bytes};")
    k.emit(f"add.s64 {addr}, {base}, {offset};")
    return "global", addr


# ---------------------------------------------------------------- leaf statements
# Self-contained: read the symbol tables, emit their own lines, define names in k.env.

VECTOR_WIDTH = {1: "", 2: ".v2", 4: ".v4"}


def emit_Load(k, s):
    """names = input[index]: one ld.global/ld.shared; several names make a vector load.

    a, b, c, d = load x[i]  ->  ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [addr];
    """
    dtype = s.dtype.name if s.dtype else "f32"
    if dtype != "f32":
        raise NotImplementedError(f"Load of {dtype}")
    space, addr = emit_address(k, s.input, s.index, 4)
    regs = [k.fresh("f") for _ in s.names]
    dst = regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"
    k.emit(f"ld.{space}{VECTOR_WIDTH[len(regs)]}.{dtype} {dst}, [{addr}];")
    k.env.update(zip(s.names, regs))


def emit_Assign(k, s, dest=None):
    """name = op(args): args are SSA names already defined (by Load, Assign, Let, ...).

    f32 v0 = add(in0, in1)  ->  add.f32 %f3, %f1, %f2;   (instruction from ELEMENTWISE)
    dest: write into this existing register instead of a fresh one (loop-carried state).
    """
    dtype = s.dtype.name if s.dtype else "f32"
    srcs = []
    for name in s.args:
        src = k.env[name]
        if not src.startswith("%"):            # a Let-bound immediate: materialize it
            reg = k.fresh("f")
            k.emit(f"mov.f32 {reg}, {src};")
            src = reg
        srcs.append(src)
    op = s.op.name

    # comparisons and where: predicates, not floats (the IR types the mask as bool)
    if op in COMPARE:
        p = k.fresh("p")
        k.emit(f"setp.{COMPARE[op]}.f32 {p}, {srcs[0]}, {srcs[1]};")
        if dtype == "bool":
            k.env[s.name] = p
            return
        d = dest or k.fresh("f")                # f32 mask convention: 1.0 / 0.0
        k.emit(f"selp.f32 {d}, 0f3F800000, 0f00000000, {p};")
        k.env[s.name] = d
        return
    if op == "where":
        cond, a, b = srcs
        if not cond.startswith("%p"):           # an f32 mask: nonzero means true
            p = k.fresh("p")
            k.emit(f"setp.ne.f32 {p}, {cond}, 0f00000000;")
            cond = p
        d = dest or k.fresh("f")
        k.emit(f"selp.f32 {d}, {a}, {b}, {cond};")
        k.env[s.name] = d
        return

    d = dest or k.fresh("f")
    for line in select_elementwise(op, dtype, d, srcs):
        k.emit(line)
    k.env[s.name] = d


def emit_Write(k, s):
    """output[index] = values: the mirror of Load; defines no names.

    add[i] = (v0, v1)  ->  st.global.v2.f32 [addr], {%f3, %f6};
    atomic writes accumulate instead: red.global.add.f32 [addr], %f;
    """
    dtype = s.value_dtype.name if s.value_dtype else "f32"
    if dtype != "f32":
        raise NotImplementedError(f"Write of {dtype}")
    space, addr = emit_address(k, s.output, s.index, 4)
    regs = [k.env[name] for name in s.values]
    if s.atomic:
        if space != "global" or len(regs) != 1:
            raise NotImplementedError("atomic Write supports one value to global memory")
        k.emit(f"red.global.add.{dtype} [{addr}], {regs[0]};")
        return
    src = regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"
    k.emit(f"st.{space}{VECTOR_WIDTH[len(regs)]}.{dtype} [{addr}], {src};")


SMEM_ELEM_BYTES = {"float": 4}


def emit_Smem(k, s):
    """Declare `.shared .align 4 .b8 name[bytes];` in k.shared and record it in k.smem.

    Declared at module scope (above .entry); the IR name doubles as the PTX symbol, so
    emit_address can take its address with `mov.u32 %r, name;`.
    """
    if s.name in k.smem:                 # declared once even if an unrolled loop revisits it
        return
    if s.dtype not in SMEM_ELEM_BYTES:
        raise NotImplementedError(f"Smem {s.name} is {s.dtype}; only float shared arrays are lowered")
    nbytes = SMEM_ELEM_BYTES[s.dtype]
    for extent in s.extents:
        nbytes *= extent
    k.smem[s.name] = s.extents
    if k.smem_pool_bytes > STATIC_SMEM_LIMIT:
        # pool mode (m01: 72 KiB): one extern array, arrays carved at aligned offsets in IR
        # order -- the same layout as emmy's _smem_pool, matching meta.json's smem_bytes
        if not k.shared:
            k.shared.append(f".extern .shared .align 16 .b8 {SMEM_POOL}[];")
        align = max(s.align, 4)
        offset = -(-k.smem_pool_used // align) * align
        k.smem_offset[s.name] = offset
        k.smem_pool_used = offset + nbytes
        return
    k.shared.append(f".shared .align {max(s.align, 4)} .b8 {s.name}[{nbytes}];")


def total_smem_bytes(stmts, used=0):
    """Bytes of every Smem declaration in the body, laid out with the same alignment padding
    emit_Smem uses (decides static vs pool mode)."""
    for s in stmts:
        if type(s).__name__ == "Smem":
            nbytes = SMEM_ELEM_BYTES.get(s.dtype, 0)   # emit_Smem reports a non-float dtype
            for extent in s.extents:
                nbytes *= extent
            align = max(s.align, 4)
            used = -(-used // align) * align + nbytes
        for key in ("body", "else_body"):
            used = total_smem_bytes(getattr(s, key, ()) or (), used)
    return used


def emit_Sync(k, s):
    """CTA barrier (__syncthreads -> bar.sync id[, count]) or warp barrier (__syncwarp)."""
    if s.warp:
        k.emit("bar.warp.sync -1;")
    elif s.count is not None:
        k.emit(f"bar.sync {s.barrier_id}, {s.count};")
    else:
        k.emit(f"bar.sync {s.barrier_id};")


# An accumulator starts at its op's identity (op.identity: add 0, multiply 1, maximum -1e30,
# minimum 1e30). emmy seeds max with -1e30 rather than -inf so that online-softmax rescaling
# (acc - new_max) never computes (-inf) - (-inf) = NaN.


def emit_Accum(k, s):
    """In-place fold: the enclosing loop (or an Init) already set k.env[s.name]; emit e.g. add.f32 acc, acc, val.

    With a base the old accumulator is already folded into it (f02: base = acc1 * alpha),
    so the update is acc = op(base, value) instead of op(acc, value).
    """
    first = s.base or s.name
    fold = Assign(name=s.name, op=s.op, args=(first, s.value), dtype=s.dtype)
    emit_Assign(k, fold, dest=k.env[s.name])


def emit_Let(k, s):
    """name = expr: a literal (f32 one = 1.0) or a precomputed index; binds, emits nothing itself.

    A float literal stays an immediate in k.env; emit_Assign materializes it on first use.
    """
    k.env[s.name] = emit_expr(k, s.value)


def emit_Init(k, s):
    """Carried accumulator seeded at Tile level: mov.f32 acc, identity. Inner loops with
    seed=False then fold into this same register across the outer loop (m01's K-chunks)."""
    acc = k.fresh("f")
    k.emit(f"mov.f32 {acc}, {float_literal(s.identity)};")
    k.env[s.name] = acc


# cp.async: 16-byte copies bypass L1 (.cg); 4/8-byte copies must use .ca
def emit_CpAsyncCopy(k, s):
    """Async global->shared copy of nbytes: cp.async.{cg|ca}.shared.global [smem], [gmem], nbytes.

    Destination element = smem_index + lane_index (componentwise, when lane_index is set).
    """
    index = s.smem_index
    if s.lane_index is not None:
        if len(s.lane_index) != len(s.smem_index):
            raise NotImplementedError("CpAsyncCopy lane_index rank differs from smem_index")
        index = tuple(BinaryExpr("+", a, b) for a, b in zip(s.smem_index, s.lane_index))
    _, dst = emit_address(k, s.smem, index, 4)
    _, src = emit_address(k, s.src, s.src_index, 4)
    mode = "cg" if s.nbytes == 16 else "ca"
    k.emit(f"cp.async.{mode}.shared.global [{dst}], [{src}], {s.nbytes};")


def emit_CpAsyncCommit(k, s):
    k.emit("cp.async.commit_group;")


def emit_CpAsyncWait(k, s):
    """Wait until at most s.group of this thread's commit groups are in flight; the IR's
    following Sync makes the copies visible block-wide."""
    k.emit(f"cp.async.wait_group {s.group};")


# ---------------------------------------------------------------- container statements
# Emit their own "before" code, lower the children in order, then their "after" code.

def emit_body(k, stmts):
    for child in stmts:
        emit_stmt(k, child)


def emit_Tile(k, s):
    """Map the thread grid onto the Tile's cells, then lower the body once per cell.

    gid = ctaid.x * ntid.x + tid.x, decoded row-major over s.axes (last axis fastest),
    e.g. axes (a0:64, a1_co:128) -> a1_co = gid % 128, a0 = gid / 128.
    Pointwise tiles (block_threads is None) may launch more threads than cells, so they
    get a bounds guard. Reduction tiles must not: every thread has to reach bar.sync.
    """
    tid, ntid, ctaid, gid = (k.fresh("r") for _ in range(4))
    k.emit(f"mov.u32 {tid}, %tid.x;")
    k.emit(f"mov.u32 {ntid}, %ntid.x;")
    k.emit(f"mov.u32 {ctaid}, %ctaid.x;")
    k.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid};")

    # implicit names used by reductions (ptxas drops them when unused)
    lane, warp = k.fresh("r"), k.fresh("r")
    k.emit(f"and.b32 {lane}, {tid}, 31;")
    k.emit(f"shr.u32 {warp}, {tid}, 5;")
    k.env["lane"], k.env["warp"] = lane, warp

    # axis i = (gid / product of later extents) % extent_i; the first axis needs no %
    extents = [axis.extent.value for axis in s.axes]
    stride = 1
    for i in reversed(range(len(s.axes))):
        value = gid
        if stride > 1:
            value, src = k.fresh("r"), value
            k.emit(f"div.u32 {value}, {src}, {stride};")
        if i > 0:
            value, src = k.fresh("r"), value
            k.emit(f"rem.u32 {value}, {src}, {extents[i]};")
        k.env[s.axes[i].name] = value
        stride *= extents[i]
    emit_raster(k, s, gid, extents)

    guarded = s.block_threads is None
    end = k.fresh_label("TILE_END")
    if guarded:
        out_of_range = k.fresh("p")
        k.emit(f"setp.ge.s32 {out_of_range}, {gid}, {stride};")  # stride == total cells
        k.emit(f"@{out_of_range} bra {end};")
    emit_body(k, s.body)
    if guarded:
        k.emit(f"{end}:")


def emit_raster(k, s, gid, extents):
    """Grouped rasterization of two adjacent block axes (a02): rebind them so consecutive
    blocks walk a raster_group-wide strip of the grouped axis, for L2 reuse.

    Any bijection of blocks to (m, n) tiles is correct -- this only changes which block gets
    which tile -- so the row-major decode already emitted stays as the fallback. Formula follows
    emmy's renderer for the divisible case (extent % group == 0):
        sub = gid / stride(n)        [% (Em*En) unless the pair is outermost]
        grouped = (sub / (g*En)) * g + sub % g
        other   = (sub % (g*En)) / g
    """
    if not s.raster_group or not s.raster_axes:
        return
    names = [a.name for a in s.axes]
    if s.raster_axes[0] not in names or s.raster_axes[1] not in names:
        return
    mi, ni = names.index(s.raster_axes[0]), names.index(s.raster_axes[1])
    if ni != mi + 1:
        return
    gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
    g = min(s.raster_group, extents[gi])
    if extents[gi] <= 1 or g <= 1 or extents[gi] % g != 0:
        return
    ea, eb = extents[gi], extents[oi]
    stride_n = 1
    for e in extents[ni + 1:]:
        stride_n *= e

    sub = gid
    if stride_n > 1:
        sub = k.fresh("r")
        k.emit(f"div.u32 {sub}, {gid}, {stride_n};")
    if mi != 0:
        sub, src = k.fresh("r"), sub
        k.emit(f"rem.u32 {sub}, {src}, {ea * eb};")
    grp, grouped, rem_g, other = (k.fresh("r") for _ in range(4))
    k.emit(f"div.u32 {grp}, {sub}, {g * eb};")
    k.emit(f"rem.u32 {rem_g}, {sub}, {g};")
    k.emit(f"mad.lo.s32 {grouped}, {grp}, {g}, {rem_g};")
    k.emit(f"rem.u32 {other}, {sub}, {g * eb};")
    k.emit(f"div.u32 {other}, {other}, {g};")
    k.env[names[gi]], k.env[names[oi]] = grouped, other


def emit_Loop(k, s):
    """for axis in 0..extent: a StridedLoop with start 0, step 1 (end defaults to the extent)."""
    emit_StridedLoop(k, StridedLoop(axis=s.axis, start=Literal(0, "int"), step=Literal(1, "int"),
                                    body=s.body, unroll=s.unroll, end=None, seed=s.seed))


# Largest unrolled body (iterations x statements) worth expanding. Measured on the runner:
#   product   768 (m01 _ki)   111 -> 42 us      2176 (a12 outer)  38 -> 32 us,  1.2 s compile
#   product  1600 (a10 outer)  42 -> 34 us      8064 (a13 outer)  no change,    7.4 s compile
#   product  1984 (m14 _ki)    no change        8512 (a01 outer)  100 -> 1277 us, 22 s compile
# Wins stop and compile time climbs past a few thousand; the cliff is before ~8k.
UNROLL_LIMIT = 4096


def count_stmts(stmts):
    """Statements in a body, nested loops and branches included."""
    n = 0
    for s in stmts:
        n += 1
        for key in ("body", "else_body", "combine_states"):
            n += count_stmts(getattr(s, key, ()) or ())
    return n


def emit_StridedLoop(k, s):
    """before: init the Accums found in s.body (look-ahead), axis = start, LOOP label, exit test.
    children: s.body.  after: axis += step, bra LOOP, END label.

    Top-tested (while-style) because a thread whose start >= end must run zero iterations.
    unroll with constant bounds is expanded at compile time: the axis becomes an immediate,
    so ptxas can fold every shared-memory address in the body (matmul's _ki loop).
    """
    # loop-carried state: each Accum gets one register, seeded BEFORE the label and then
    # updated in place by emit_Accum (a mutable register instead of an SSA phi).
    # seed=False (matmul's inner _ki loop): the accumulators were Init'ed outside and must
    # keep folding across the outer loop, so nothing is re-seeded here.
    for child in s.body:
        if s.seed and type(child).__name__ == "Accum":
            acc = k.fresh("f")
            k.emit(f"mov.f32 {acc}, {float_literal(child.op.identity)};")
            k.env[child.name] = acc

    start = emit_expr(k, s.start)
    end = emit_expr(k, s.end) if s.end is not None else str(s.axis.extent.value)
    step = emit_expr(k, s.step)
    if (s.unroll and all(v.isdigit() for v in (start, end, step))
            and len(range(int(start), int(end), int(step))) * count_stmts(s.body) <= UNROLL_LIMIT):
        for value in range(int(start), int(end), int(step)):
            k.env[s.axis.name] = str(value)
            emit_body(k, s.body)
        return

    # the loop variable is a copy: start (e.g. a0_co) is still read after the loop
    i = k.fresh("r")
    k.emit(f"mov.s32 {i}, {start};")
    k.env[s.axis.name] = i

    loop, done = k.fresh_label("LOOP"), k.fresh_label("LOOP_END")
    finished = k.fresh("p")
    k.emit(f"{loop}:")
    k.emit(f"setp.ge.s32 {finished}, {i}, {end};")
    k.emit(f"@{finished} bra {done};")
    emit_body(k, s.body)
    k.emit(f"add.s32 {i}, {i}, {step};")
    k.emit(f"bra {loop};")
    k.emit(f"{done}:")


def emit_Cond(k, s):
    """before: setp from emit_expr(s.cond), branch to ELSE.  children: s.body, then s.else_body.
    after: END label.

    Lowered as a branch around the body (if !cond goto ELSE/END), like the loop minus the
    back-edge. Comparisons already yield a %p; any other int value is tested against 0.
    """
    cond = emit_expr(k, s.cond)
    if not cond.startswith("%p"):
        pred = k.fresh("p")
        k.emit(f"setp.ne.s32 {pred}, {in_register(k, cond)}, 0;")
        cond = pred

    done = k.fresh_label("IF_END")
    if not s.else_body:
        k.emit(f"@!{cond} bra {done};")
        emit_body(k, s.body)
    else:
        other = k.fresh_label("ELSE")
        k.emit(f"@!{cond} bra {other};")
        emit_body(k, s.body)
        k.emit(f"bra {done};")
        k.emit(f"{other}:")
        emit_body(k, s.else_body)
    k.emit(f"{done}:")


def emit_WarpShuffle(k, s):
    """for offset in 16,8,4,2,1: shfl.sync.bfly each state into its state_b name,
    then emit_body(k, s.combine_states).

    XOR butterfly: after log2(length) steps every lane holds the combined value. Unrolled at
    compile time, so the code is straight-line and combine_states may rebind names to fresh
    registers. All states are shuffled before any combine, so partners see the old values.
    The full member mask is safe because reduction tiles have no bounds guard.
    """
    offset = s.length // 2
    while offset >= 1:
        for mine, partner in zip(s.state, s.state_b):
            bits, moved, value = k.fresh("r"), k.fresh("r"), k.fresh("f")
            k.emit(f"mov.b32 {bits}, {k.env[mine]};")       # shfl moves bits, not floats
            k.emit(f"shfl.sync.bfly.b32 {moved}, {bits}, {offset}, 31, -1;")
            k.emit(f"mov.b32 {value}, {moved};")
            k.env[partner] = value
        emit_body(k, s.combine_states)
        offset //= 2


def emit_TreeHalve(k, s):
    """Follow the .ref.cu loop: per level load both halves from smem, emit_body(combine_states),
    store back, bar.sync; finally every thread reads smem[0] (broadcast).

    Desugared into IR the other handlers already lower, one level per halving step:
        if (warp < h) { state = buf[warp]; state_b = buf[warp + h]; combine; buf[warp] = state; }
        bar.sync                      <- outside the if: every thread must arrive
    then state = buf[0] in every thread. Levels are unrolled at compile time (length is constant).
    """
    if s.inner is not None:
        raise NotImplementedError("TreeHalve with inner not supported yet")
    tid = Var(s.tid_var)
    half = s.length // 2
    while half >= 1:
        body = []
        for buf, mine, partner in zip(s.bufs, s.state, s.state_b):
            body.append(Load(names=(mine,), input=buf, index=(tid,), dtype=s.dtype))
            body.append(Load(names=(partner,), input=buf,
                             index=(BinaryExpr("+", tid, Literal(half, "int")),), dtype=s.dtype))
        body.extend(s.combine_states)
        for buf, mine in zip(s.bufs, s.state):
            body.append(Write(output=buf, index=(tid,), values=(mine,), value_dtype=s.dtype,
                              atomic=False, swizzle="NONE"))
        emit_stmt(k, Cond(cond=BinaryExpr("<", tid, Literal(half, "int")), body=tuple(body)))
        emit_stmt(k, Sync(barrier_id=s.barrier_id, count=s.barrier_count))
        half //= 2
    for buf, mine in zip(s.bufs, s.state):
        emit_stmt(k, Load(names=(mine,), input=buf, index=(Literal(0, "int"),), dtype=s.dtype))


# ---------------------------------------------------------------- dispatch

LEAF_STMTS = {
    "Load": emit_Load, "Assign": emit_Assign, "Write": emit_Write,
    "Smem": emit_Smem, "Sync": emit_Sync, "Accum": emit_Accum,
    "Let": emit_Let, "Init": emit_Init,
    "CpAsyncCopy": emit_CpAsyncCopy, "CpAsyncCommit": emit_CpAsyncCommit,
    "CpAsyncWait": emit_CpAsyncWait,
}
CONTAINER_STMTS = {
    "Tile": emit_Tile, "StridedLoop": emit_StridedLoop, "Loop": emit_Loop, "Cond": emit_Cond,
    "WarpShuffle": emit_WarpShuffle, "TreeHalve": emit_TreeHalve,
}
HANDLERS = {**LEAF_STMTS, **CONTAINER_STMTS}


def emit_stmt(k, s):
    """Lower one statement by dispatching on its IR class name."""
    kind = type(s).__name__
    if kind not in HANDLERS:
        raise NotImplementedError(f"no lowering for IR statement {kind}")
    k.emit(f"// IR: {kind}")
    HANDLERS[kind](k, s)


def main():
    ir_path = sys.argv[1]
    op = load_kernel(ir_path)
    k = PtxKernel(op.name, load_arg_order(ir_path, op.name))
    k.shapes, k.dtypes = load_buffer_shapes(ir_path, op.name)
    k.smem_pool_bytes = total_smem_bytes(op.body)   # static arrays, or one dynamic pool
    k.smem_pool_used = 0
    emit_prologue(k)
    emit_body(k, op.body)
    k.emit("ret;")
    print(k.to_text())


if __name__ == "__main__":
    main()
