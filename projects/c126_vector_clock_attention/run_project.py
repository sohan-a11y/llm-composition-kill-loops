"""
Autonomous Project Runner for C126 (Vector-Clock Asynchronous Causal Attention)
=============================================================================
Executes full multi-seed evaluation, verifies all gating thresholds, and outputs
structured results. Returns exit code 0 on success.
"""

import sys
import os

# Add project root to sys.path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)

from evaluate import run_full_evaluation


def main():
    log_dir = os.path.join(curr_dir, "logs")
    summary = run_full_evaluation(seeds=[42, 137], output_dir=log_dir)

    if not summary["overall_pass"]:
        print("\n[ERROR] C126 failed empirical validation gates!")
        sys.exit(1)

    print("\n[SUCCESS] C126 passed all empirical validation gates!")
    sys.exit(0)


if __name__ == "__main__":
    main()
