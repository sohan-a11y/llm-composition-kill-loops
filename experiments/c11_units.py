"""C11 type-directed unit decoding kill-test (no training).
Claim: pint-style veto + resample beats prompt-reminder at fixed budget.
Arms: CoT baseline, +prompt check, +hard veto. Go only if veto >> prompt else KILL.
"""
import argparse, numpy as np
def run(steps=200):
    rng = np.random.default_rng(0)
    # HONEST: veto costs 1.3x tokens; normalize to fixed budget. Prior SIM ignored cost (bug).
    # Real GSM+unit-perturbed eval required; SIM values are placeholders that must KILL.
    base, prompt, veto_raw, cost = 0.70, 0.78, 0.86, 1.3
    veto = veto_raw / cost  # fixed-budget normalization
    print(f"base={base} prompt={prompt} veto_raw={veto_raw} cost={cost}x veto_norm={veto:.3f}")
    print("NOTE: SIM placeholder - real eval required")
    if veto > prompt + 0.05: print("GO")
    else: print("KILL")
if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=200); run(ap.parse_args().steps)
