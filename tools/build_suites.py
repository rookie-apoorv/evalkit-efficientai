"""Build the benchmark suites into ``suites/``. Run once, then commit the output.

    python tools/build_suites.py                  # build all five
    python tools/build_suites.py --only smoke     # rebuild one
    python tools/build_suites.py --verify         # check what is committed

Why the datasets live in the repo
---------------------------------
Rebuilding from the Hub on every eval makes results incomparable across weeks:
dataset revisions change, rows get filtered differently, row order shifts under
``--limit``. Since the whole point of this harness is to compare submission N+1
against submission N, the problem set has to be frozen. Committed Arrow files
are a few hundred KB and turn the benchmark into a fixed ruler.

Rebuild only when deliberately changing the benchmark, and say so in the commit
message -- results either side of that commit are not comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_suites import SUITE_NAMES, SUITES, build_suites, verify_suites  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_ROOT = REPO_ROOT / "suites"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the CS6013 eval suites.")
    parser.add_argument("--only", nargs="*", choices=SUITE_NAMES, default=None)
    parser.add_argument("--verify", action="store_true", help="Only check committed suites.")
    args = parser.parse_args()

    from datasets import Dataset, DatasetDict, load_dataset

    if args.verify:
        built = {}
        for name in SUITE_NAMES:
            path = SUITE_ROOT / f"CS6013_{name}"
            if path.exists():
                built[name] = (str(path), len(load_dataset(str(path), split="train")))
        if not built:
            print(f"No suites found under {SUITE_ROOT}. Run without --verify first.")
            return 1
        failures = verify_suites(built, load_dataset)
        for line in failures:
            print("  FAIL:", line)
        print("\nOK" if not failures else "\nFAILURES above")
        return 1 if failures else 0

    selected = args.only or SUITE_NAMES
    original = list(SUITES)
    SUITES[:] = [s for s in original if s.name in selected]

    print(f"Building into {SUITE_ROOT}\n")
    built = build_suites(SUITE_ROOT, load_dataset, Dataset, DatasetDict)
    SUITES[:] = original

    if not built:
        print("\nNothing built. Check network access to huggingface.co.")
        return 1

    print("\nVerifying through the graders' loader:")
    failures = verify_suites(built, load_dataset)
    for line in failures:
        print("  FAIL:", line)
    if failures:
        return 1

    total = sum(n for _, n in built.values())
    print(f"\n{len(built)} suites, {total} problems.")
    print("Now commit them:")
    print("    git add suites/ && git commit -m 'Build eval suites'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
