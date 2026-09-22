# Project C131: Self-Stabilizing Dijkstra-Ring Virtual Token Memory

## 1. Mathematical Mechanism & Hypothesis

### Theoretical Background
In neural models with persistent working memory slots, state-space models, or recurrent scratchpads, transient faults—such as SRAM soft-errors from cosmic radiation, extreme float quantization truncation, or adversarial token perturbations—frequently corrupt internal memory slots, trapping the model in permanent hallucination loops.

In 1974, Edsger W. Dijkstra introduced the concept of **Self-Stabilization** in distributed computing:
A system is self-stabilizing if, regardless of its initial state (including arbitrary corrupted memory, deadlocks, or multi-privilege states), it is mathematically guaranteed to converge to a legitimate configuration within a finite number of transitions without external intervention.

### The Dijkstra Token Ring for Working Memory
We structure $K$ working memory slots into a Dijkstra Token Ring where each slot $i \in \{0, \dots, K-1\}$ maintains a discrete state $s_i \in \mathbb{Z}_M$ ($M \ge K$):

1. **Node 0 (Leader Privilege Rule):**
   Node 0 has privilege iff:
   $$s_0 = s_{K-1}$$
   State transition when privileged:
   $$s_0 \leftarrow (s_{K-1} + 1) \pmod M$$

2. **Worker Nodes (Node $i > 0$ Privilege Rule):**
   Node $i$ has privilege iff:
   $$s_i \ne s_{i-1}$$
   State transition when privileged:
   $$s_i \leftarrow s_{i-1}$$

### Dijkstra's Self-Stabilization Theorem (1974)
From **any arbitrary initial state** $s \in \mathbb{Z}_M^K$ (all $M^K$ possible configurations, including illegal states with 0 privileges or multiple concurrent privileges), the system converges to a legitimate configuration having **exactly one privilege** ($P = 1$) circulating perpetually in at most $\mathcal{O}(K^2)$ steps (empirically $\le 5$ steps for $K=4, M=5$, mean $2.04$ steps across all 625 configurations).

In this architecture, memory slots receive write updates gated by the privilege vector $\mathbf{priv} \in \{0, 1\}^K$. When transient corruption strikes, the ring autonomously purges spurious privileges and self-stabilizes back to mutual exclusion ($P=1$), preserving long-term memory integrity without external reboot.

---

## 2. Experimental Design & Falsification Arms

### Task: Sequence State Tracking Under Hardware Corrupted States
- Sequence length: $T = 6$ steps.
- $K = 4$ memory slots, $M = 5$ states, $5$ target classes.
- **Sanity Check:** Clean execution initialized from legitimate state $s = [0, 0, 0, 0]$ (Threshold: $\ge 0.85$).
- **Multi-Seed Fault Tolerance Check:** Model is initialized in arbitrary uniform random state configurations $s \sim \text{Uniform}(0, M-1)^K$. Evaluated across seeds `[42, 137]`.
- **Mutex Convergence Rate:** Proportion of sequences where the ring converges to exactly 1 privilege ($P = 1$) by step 5 (Threshold: $\ge 0.95$).
- **Lethal Negative Control 1 (Deadlock Anti-Dijkstra Mode):** Freezes the leader token when $s_0 = s_{K-1}$, permanently creating 0 privileges. Accuracy must collapse to chance baseline ($1/5 = 0.2000$).
- **Lethal Negative Control 2 (Scramble Mode):** Randomly permutes ring node assignments at each step, destroying spatial invariants. Accuracy must collapse to chance baseline ($0.2000$).

---

## 3. Empirical Results Summary

| Metric | Threshold | Seed 42 | Seed 137 | Mean $\pm$ Std | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sanity Check (Clean Execution)** | $\ge 0.85$ | 1.0000 | 1.0000 | **1.0000 $\pm$ 0.0000** | **PASS** |
| **Fault Tolerance (Corrupted States)** | $\ge 0.90$ | 1.0000 | 1.0000 | **1.0000 $\pm$ 0.0000** | **PASS** |
| **Mutex Convergence Rate ($P=1$)** | $\ge 0.95$ | 1.0000 | 1.0000 | **1.0000 $\pm$ 0.0000** | **PASS** |
| **Kill Control 1 (Deadlock Mode)** | $\le 0.30$ | 0.1840 | 0.1800 | **0.1820** (Chance: 0.2000) | **COLLAPSED** |
| **Kill Control 2 (Scramble Mode)** | $\le 0.30$ | 0.1840 | 0.1800 | **0.1820** (Chance: 0.2000) | **COLLAPSED** |

---

## 4. Reproduction & Execution
```bash
python projects/c131_dijkstra_ring/run_project.py
```
