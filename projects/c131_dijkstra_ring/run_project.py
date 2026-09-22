"""
Autonomous Project Runner for C131: Self-Stabilizing Dijkstra-Ring Virtual Token Memory
=====================================================================================
Executes full multi-seed evaluation across seeds [42, 137], verifies fault-tolerance
and lethal negative controls, and returns exit code 0 on success.
"""

import sys
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)

from evaluate import run_full_evaluation


def main():
    log_dir = os.path.join(curr_dir, "logs")
    summary = run_full_evaluation(seeds=[42, 137], output_dir=log_dir)

    if not summary["overall_pass"]:
        print("\n[ERROR] C131 failed empirical validation gates!")
        sys.exit(1)

    print("\n[SUCCESS] C131 passed all empirical validation gates!")
    sys.exit(0)


if __name__ == "__main__":
    main()
