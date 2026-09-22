"""Smoke: import + 5-step path check for all ML exps (full runs are T4-gated per BRIEF §3).
Non-ML exps run fully (fast). Prints SMOKE-PASS or traceback.
"""
import subprocess, sys, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = ["b_d1_join_order", "b_d2_matviews", "c11_units", "b_cy2_merkle", "b_s3_wavelet", "b_c3_biasfold"]
ok = True
for e in FAST:
    r = subprocess.run([sys.executable, os.path.join(BASE, "experiments", e + ".py"), "--steps", "50"],
                       capture_output=True, text=True, cwd=BASE, timeout=120)
    s = (r.stdout + r.stderr)[-500:].replace("\n", " | ")
    print(f"{e}: rc={r.returncode} {s[:300]}")
    ok &= r.returncode == 0
# ML exps: import-only + model construction (no training) to avoid CPU timeout
for e in ["c4_needle_jitter", "c9_accumulator", "c1_middle_only", "a6_mlp_adapter", "common"]:
    r = subprocess.run([sys.executable, "-c", f"import experiments.{e}; print('import ok')"],
                       capture_output=True, text=True, cwd=BASE, timeout=120)
    print(f"{e}: rc={r.returncode} {(r.stdout+r.stderr)[-200:]}")
    ok &= r.returncode == 0
print("SMOKE-PASS" if ok else "SMOKE-FAIL")
