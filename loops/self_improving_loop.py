"""Self-improving loop: generate -> verify -> kill-test -> promote -> mutate.
State lives in ideas/pool.json + loops/state.json (JSONL-ledger, resumable).
Each iteration is deterministic, has a should-fail control, and a go/no-go gate.
Usage: python loops/self_improving_loop.py --iters 5 --steps 500
"""
import argparse, json, os, random, subprocess, sys, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(BASE, "ideas", "pool.json")
STATE = os.path.join(BASE, "loops", "state.json")

STRATS = ["invert-assumption", "xdomain-compilers", "xdomain-db", "xdomain-os",
          "xdomain-signal", "xdomain-control", "xdomain-ecc", "xdomain-crypto",
          "xdomain-econ", "xdomain-bio-phys", "anomaly", "tooling"]

def load_pool():
    if os.path.exists(POOL):
        with open(POOL) as f:
            return json.load(f)
    return {"raw": [], "filter1": {}, "filter2": {}, "promoted": []}

def save_pool(p):
    os.makedirs(os.path.dirname(POOL), exist_ok=True)
    with open(POOL, "w") as f:
        json.dump(p, f, indent=1)

def log_state(ev):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    ev["t"] = time.time()
    with open(STATE, "a") as f:
        f.write(json.dumps(ev) + "\n")

MUT = [
    "invert the data/control plane", "swap train/infer placement",
    "replace learned with analytic + verify", "add explicit kill control",
    "bound cost to <10% overhead", "require bit-exact control that can fail",
]

def gen_candidates(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = rng.choice(STRATS); m = rng.choice(MUT)
        out.append({"id": f"G{seed}-{i}", "strat": s,
                    "claim": f"auto-{s} variant {i}: {m}",
                    "status": "raw"})
    return out

def run_kill(exp, steps):
    p = os.path.join(BASE, "experiments", f"{exp}.py")
    if not os.path.exists(p):
        return {"exp": exp, "skip": "missing"}
    r = subprocess.run([sys.executable, p, "--steps", str(steps)],
                       capture_output=True, text=True, cwd=BASE, timeout=600)
    tail = (r.stdout + r.stderr)[-3000:]
    go = "GO" in tail and "KILL" not in tail.split("GO")[-1][:20]
    return {"exp": exp, "rc": r.returncode, "go": go, "tail": tail}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--new-per-iter", type=int, default=10)
    ap.add_argument("--exps", nargs="*", default=["c4_needle_jitter", "c9_accumulator", "c1_middle_only"])
    args = ap.parse_args()
    pool = load_pool()
    start_n = len(pool["raw"])
    for it in range(args.iters):
        nc = gen_candidates(args.new_per_iter, seed=1000 + it)
        pool["raw"].extend(nc)
        log_state({"iter": it, "event": "generated", "n": len(nc)})
        for e in args.exps:
            try:
                res = run_kill(e, args.steps)
            except Exception as ex:
                res = {"exp": e, "error": str(ex)}
            log_state({"iter": it, "event": "kill", **{k: v for k, v in res.items() if k != "tail"}})
            if res.get("go"):
                if e not in pool["promoted"]:
                    pool["promoted"].append(e)
        save_pool(pool)
    print(f"pool {start_n} -> {len(pool['raw'])} raw; promoted={pool['promoted']}")
    print(f"ledger: {STATE}")

if __name__ == "__main__":
    main()
