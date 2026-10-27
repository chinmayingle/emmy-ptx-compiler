"""Run compile.py on every open problem and print a summary table (for the demo).

    .venv/bin/python run_all.py              # compile + run on the hosted runner (uses submit.py)
    .venv/bin/python run_all.py --local      # compile only, no network: checks that we emit PTX
    .venv/bin/python run_all.py p01_vector_add r01_array_sum   # a subset
    .venv/bin/python run_all.py -v           # also show each problem's full output
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
CATEGORIES = {"p": "pointwise", "r": "reductions", "m": "matmul", "f": "fused", "a": "attention"}


def run_remote(problem):
    """submit.py compiles with our compiler, ptxas-assembles, runs on the GPU and checks outputs."""
    r = subprocess.run([sys.executable, "submit.py", problem, "--compiler", "compile.py"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def run_local(problem):
    """Compile every kernel of the problem; success means PTX came out, not that it is correct."""
    lines = []
    for ir in sorted((ROOT / "problems" / problem).glob("*.ir.json")):
        r = subprocess.run([sys.executable, "compile.py", str(ir)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            err = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            return False, f"{ir.name}: compile failed: {err}"
        lines.append(f"{ir.name}: {len(r.stdout.splitlines())} lines of PTX")
    return True, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="*", help="problem ids (default: all)")
    ap.add_argument("--local", action="store_true", help="compile only; do not call the runner")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each problem's output")
    args = ap.parse_args()

    names = args.problems or sorted(p.name for p in (ROOT / "problems").iterdir() if p.is_dir())
    run = run_local if args.local else run_remote
    results = []
    for name in names:
        start = time.time()
        ok, out = run(name)
        results.append((name, ok, time.time() - start, out))
        print(f"{'PASS' if ok else 'FAIL'}  {name:<20} {results[-1][2]:5.1f}s", flush=True)
        if args.verbose or not ok:
            print("      " + out.replace("\n", "\n      "))

    print("\nsummary by category" + (" (compile only)" if args.local else ""))
    for letter, label in CATEGORIES.items():
        group = [r for r in results if r[0].startswith(letter)]
        if group:
            passed = sum(r[1] for r in group)
            print(f"  {label:<11} {passed}/{len(group)}")
    total = sum(r[1] for r in results)
    print(f"  {'total':<11} {total}/{len(results)}")
    return 0 if total == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
