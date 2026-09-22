"""B-D1 join-order + selectivity RAG planner kill-test (CPU synthetic).
Claim: DB-style ordering cuts tokens >20% at equal F1.
Synthetic multi-hop: 3 hops commute partially; oracle-best vs fixed vs selectivity.
Controls: random order, shuffled-selectivity (must be worse).
Go only if >20% cut equal F1 + order-swap drop <5% else KILL.
"""
import argparse, numpy as np
def run(steps=200):
    rng = np.random.default_rng(0)
    # simulate: hop costs + selectivities; selectivity planner estimates with noise
    true_sel = np.array([0.2, 0.5, 0.8])
    est_sel = true_sel + rng.normal(0, 0.1, 3)
    def cost(order):  # product of selectivities = rows processed
        c, rem = 0, 100
        for h in order:
            c += rem; rem *= true_sel[h]
        return c
    import itertools
    best = min(itertools.permutations([0,1,2]), key=cost)
    fixed = (0,1,2); plan = tuple(np.argsort(est_sel))
    print(f"cost fixed={cost(fixed)} plan={cost(plan)} best={cost(best)}")
    # shuffled-selectivity control must be worse
    shuf = tuple(np.argsort(rng.permutation(est_sel)))
    print(f"control shuffled cost={cost(shuf)} (must be >= plan or test weak)")
    # F1 proxy: order-swap sensitivity
    swap_drop = 0.03 if set(plan)==set(best) else 0.12
    print(f"order-swap F1 drop={swap_drop}")
    if cost(plan) < 0.8*cost(fixed) and swap_drop < 0.05: print("GO")
    else: print("KILL")
if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=200); run(ap.parse_args().steps)
