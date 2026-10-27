# Kernel IR -> PTX compiler (Emmy Hackathon 2026 submission)

**Author:** Chinmay Ingle (chinmay.r.ingle@gmail.com)

`compile.py` lowers Emmy's Kernel IR straight to `sm_89` PTX, in one Python file with no
dependency beyond `emmy-ml` (to deserialize the IR).

```bash
pip install emmy-ml==0.3.11 numpy requests
python compile.py problems/m01_matmul_512/k_matmul_2967be.ir.json > out.ptx
python submit.py m01_matmul_512 --compiler compile.py     # one problem on the hosted runner
python run_all.py                                          # all 15, as a table
```

The `problems/`, `submit.py`, `tools/` and `docs/` folders are the organizers' task repository,
included so the compiler can be run as the hackathon evaluates it. The original task README
follows below the submission notes.

## Results

All 15 open problems pass on the hosted RTX 5090 runner, within each problem's tolerance:

| category | problems | result |
|---|---|---|
| pointwise | p00, p01, p02, p03 | 4 / 4 |
| reductions | r01, r02, r03, r04 | 4 / 4 |
| matmul | m01, m02 | 2 / 2 |
| fused | f01, f02, f03 | 3 / 3 |
| attention | a01, a02 | 2 / 2 |

## How the compiler works

Every IR node is lowered by one function, and they fall into three layers:

- **Pattern layer** (table-driven): `ELEMENTWISE` and `INDEX_OPS` map an op to a PTX template
  (`add` -> `add.f32`; `exp` -> multiply by log2(e) then `ex2.approx`, since PTX has no base-e
  exp). `Load`, `Assign`, `Write`, `Let`, `Init`, `Sync` and the `cp.async` trio are leaf handlers.
- **Logic layer**: `Tile`, `StridedLoop`/`Loop`, `Cond`, `WarpShuffle`, `TreeHalve` create control
  flow or loop-carried state, which no pattern can express. Rule of thumb: if lowering needs a
  label, a register that outlives the statement, or whole-kernel knowledge, it is not a pattern.
- **ABI and layout**: the prologue reads parameters in `meta.json` `arg_order` and converts them
  to global addresses; `flat_index` flattens multi-dim indices against buffer shapes; `emit_Smem`
  lays out shared memory; `to_text` declares registers.

Registers are virtual: every SSA name gets a fresh `%f`/`%r`/`%rd`, counts are declared at the
top, and `ptxas` does the real allocation. There is no liveness analysis.

### The parts that were hard

- **Loop-carried accumulators.** `Accum` is the IR's phi node, lowered as a mutable register: seeded
  with `op.identity` before the loop label and updated in place. The IR's `seed` flag must be
  honored: matmul's inner K loop has `seed=False` because its accumulators were `Init`ed at tile
  level and carry across the outer K-chunk loop. Re-seeding assembles fine and keeps only the last
  chunk.
- **The guard the IR omits.** Pointwise tiles get `gid < N` derived from the Tile; cooperative tiles
  must not be guarded, or a thread that skips the body skips `bar.sync` and the block deadlocks.
  Verified against emmy's own rule (`n % cells != 0`).
- **Reductions across threads.** `WarpShuffle` is an XOR butterfly unrolled at compile time;
  `TreeHalve` is rewritten into `Cond`/`Load`/`Write`/`Sync` nodes the compiler already lowers, one level
  per halving, with the barrier outside the `if`.
- **cp.async and the 48 KiB wall.** `cp.async` copies global -> shared with no register and no wait;
  `commit_group`/`wait_group` plus the IR's `bar.sync` make a 3-stage pipeline. The 512x512 matmul
  needs 72 KiB of shared memory, above the 48 KiB static cap, so the compiler switches to one
  `.extern .shared` pool with aligned per-array offsets (emmy's own layout) when the total exceeds
  the cap. The switch lives inside `emit_address`, so loads, stores and cp.async are unaware of it.
- **Rasterized tiles.** Only change which block gets which output tile (L2 reuse), so any bijection
  is correct; row-major decode is the fallback and the grouped formula rebinds the two block axes.

### Limitations

- f32 buffers only. f16/i32/f64 buffers and non-float shared arrays are refused at compile time
  (the IR says f32 on a store even when the buffer is f16; `meta.json` is where it shows).
- Tensor-core nodes (`RegFragment`, `mma`, `wgmma`, TMA) and `TreeHalve.inner` are not lowered.
- `pow`, `sin`, `cos`, `erf` are not in the op table. Any unhandled node raises
  `NotImplementedError` rather than guessing.

### Other files

`run_all.py` runs the 15 open problems through `submit.py` and prints a table. Everything else
is the organizers' task repository, unchanged. An AI coding assistant
was used during development (allowed by the rules) for parts of the lowering and for review; the
design and every lowering decision are mine and are explained in the video.

---

# Original task README: Emmy Hackathon — write a PTX compiler for Kernel IR

Emmy (github.com/cloudrift-ai/emmy) compiles PyTorch to CUDA C++ and builds it with `nvcc`.
**Your task: write a compiler that takes Emmy's Kernel IR and emits PTX directly**, skipping the
CUDA C++ step. `ptxas` turns PTX into the same GPU binary ~19× faster than the full `nvcc`
pipeline — Emmy's autotuner compiles thousands of kernel variants, so that middle step matters.

You do **not** need a GPU. A hosted runner at `http://runner.riftstack.ai:48701` (the default in `submit.py`) compiles
your PTX with `ptxas`, executes it on real hardware, and tells you whether the output is correct. See [docs/API.md](docs/API.md).

## Requirements

- **Python 3.12 or newer.** The Kernel IR only deserializes through Emmy's own classes:

  ```bash
  pip install emmy-ml==0.3.11       # no torch, no CUDA, no GPU needed
  ```

  This alone reads every problem IR. Torch is **not** needed — it is only required to
  *generate* new IRs from PyTorch expressions, which is optional.

- Your submission is a **program**, not PTX files. It must run as:

  ```bash
  python compile.py path/to/kernel.ir.json > out.ptx
  ```

  This contract matters: submissions are evaluated against a **hidden set** of Kernel IRs in the
  same categories as the open problems. Hardcoding the open set scores nothing.

## Five-minute start

```bash
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 && cd emmy-hackathon-2026
pip install emmy-ml==0.3.11 numpy requests   # no torch needed
python submit.py r01_array_sum examples/k_sum_1_reduce.ptx    # see a pass
python submit.py p00_scalar_add examples/k_sum_1_reduce.ptx   # wrong entry: a clean failure
python submit.py p00_scalar_add --compiler compile.py         # your compiler, once you have one
```

Then open `problems/p00_scalar_add/` and look at three files:

| file | what it is |
|---|---|
| `*.ir.json` | the Kernel IR — **the input to your compiler** |
| `*.ir.txt` | the same IR, pretty-printed for reading |
| `*.ref.cu` | Emmy's generated CUDA C++ — the reference answer, in a different target language |

Load an IR in Python:

```python
import json
from emmy.compiler.graph import Graph
g = Graph.from_dict(json.load(open("problems/r01_array_sum/k_sum_1_reduce.ir.json")))
body = g.nodes["sum_1"].op.body     # a tree of typed statements: Tile, StridedLoop, Load, ...
```

Your compiler walks that tree and prints PTX. Start with the common node kinds in the short table —
see [TASK.md](TASK.md) for the categories and the node-by-node translation table.

## The problems

15 open problems in 5 categories, easy → hard: **pointwise → reductions → matmul → fused →
attention-like**. Each is a folder under `problems/` with the IR, the reference, a launch
descriptor (`meta.json` — you never guess grid/block), and expected outputs (`expected.npz`).

**MVP: your compiler passes categories 1 and 2.** See [MVP.md](MVP.md).

Problems are scheduled for `sm_89`, and the runner is an **RTX 5090 (`sm_120`)**, where PTX written for `sm_89`
runs unchanged; [problems/PROVENANCE.md](problems/PROVENANCE.md)
says how they were generated and checked. The hidden evaluation set contains more problems of the *same kinds* —
different shapes and ops, no surprise concepts.

## Submitting

Submit at [riftstack.ai/hackathon/submit](https://riftstack.ai/hackathon/submit) before
**23:59 AoE on Tuesday, October 27**. A submission is:

- a **public GitHub repository** with `compile.py` at its root, runnable as
  `python compile.py path/to/kernel.ir.json > out.ptx`;
- a **video of 3 to 5 minutes** in which you walk through your compiler in your own voice
  (YouTube, Vimeo or Google Drive, shared so that anyone with the link can watch it).

Teams of up to three submit once and list every member. You can submit again until the
deadline; we review your latest entry. AI coding assistants are allowed, but your video has to
show that you understand how your compiler works — an entry where the understanding seems to
have been handed to an AI is not eligible for a prize.

## How compilers are evaluated

We run your `compile.py` on every kernel of the **hidden set** — the same five categories as the
open problems, with different shapes and ops — then assemble the PTX with `ptxas`, run it on the
runner's GPU under each problem's `meta.json` launch, and compare the outputs with the expected
ones within the problem's tolerance. This is the pipeline `submit.py --compiler` runs, so passing
the open set that way is the best preparation. Keep compiling to seconds per kernel: a time
limit may apply.

1. **Correctness on the hidden set, by category.** Passing categories 1 and 2 is the bar
   ([MVP.md](MVP.md)); each further category is what sets submissions apart.
2. **Code review** of your compiler is the first tiebreak. The IR → PTX lowering has to be your
   own — see the rule in [TASK.md](TASK.md).
3. **Speed** of the generated kernels is the last tiebreak.

The video decides prize eligibility, as described above.

## Repo map

```
problems/     the open set (15 problems)
examples/     hand-written PTX with comments mapping each block back to the IR
submit.py     one-command test: compile + run + check against expected outputs
docs/API.md   the runner API, if you want to call it directly
tools/        how the problems were generated (emmy compile -c ... --ir cuda)
TASK.md       the task, the categories, the CLI contract
MVP.md        the minimum bar
```
