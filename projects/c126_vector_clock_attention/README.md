# Project C126: Vector-Clock Asynchronous Causal Attention

## 1. Mathematical Mechanism & Hypothesis

### Theoretical Background
Standard autoregressive transformers enforce a rigid **linear total order** on token positions: token $t$ attends only to tokens $\{1, \dots, t-1\}$ via a static lower-triangular causal mask:
$$M_{i,j}^{\text{total}} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

In concurrent reasoning scenarios (e.g., Tree-of-Thoughts exploration, multi-agent debate, branch-and-bound verification, or speculative decoding with parallel branches), this total ordering imposes an artificial serial bottleneck:
1. **Barrier Synchronization Overhead:** Independent branches cannot generate asynchronously without waiting for global lock barriers.
2. **Nondeterministic Causal Violations:** If branches write tokens asynchronously into a shared linear sequence, thread scheduling or network latency causes race conditions where uncommitted future tokens contaminate parallel branches.

### The Vector-Clock Causal Attention Mechanism
We map distributed systems **Lamport vector clocks** (Fidge-Mattern causality) directly into transformer self-attention.

Let $B$ be the number of concurrent asynchronous streams. Each token $x_i$ produced by stream $s_i \in \{0, \dots, B-1\}$ is assigned an integer vector clock $V(x_i) \in \mathbb{N}^B$:
1. **Local Token Emission:**
   $$V(x_i)[s_i] = V(\text{prev}_{s_i})[s_i] + 1, \quad V(x_i)[k] = V(\text{prev}_{s_i})[k] \; (k \ne s_i)$$
2. **Cross-Stream Asynchronous Message Passing:**
   When stream $s_i$ receives an intermediate finding or message token $x_m$ from stream $m$:
   $$V(x_i)[b] = \max(V(\text{prev}_{s_i})[b], V(x_m)[b]) \quad \forall b \in \{0, \dots, B-1\}, \quad V(x_i)[s_i] = V(x_i)[s_i] + 1$$

The **Vector-Clock Causal Attention Mask** $M \in \{0, -\infty\}^{N \times N}$ is governed strictly by the distributed partial order $\preceq$:
$$x_j \preceq x_i \iff \forall b \in \{0, \dots, B-1\}, \; V(x_j)[b] \le V(x_i)[b]$$
$$M_{i,j} = \begin{cases} 0.0 & \text{if } x_j \preceq x_i \\ -\infty & \text{if } x_j \not\preceq x_i \text{ (concurrent } x_i \parallel x_j \text{ or causal future } x_i \prec x_j) \end{cases}$$

### Key Properties
- **Exact Concurrency Isolation:** If two branches generate concurrently without synchronization ($x_i \parallel x_j$), their clocks are incomparable ($\exists b_1, b_2: V(x_i)[b_1] > V(x_j)[b_1]$ and $V(x_j)[b_2] > V(x_i)[b_2]$). Thus $M_{i,j} = M_{j,i} = -\infty$, preventing mutual cross-talk without locks.
- **$\mathcal{O}(1)$ Causal Merge:** When an asynchronous message is received, all causal predecessors in the sender's history instantly become visible in a single tensor operation.
- **Single-Stream Equivalence:** For $B=1$, $V(x_i) = [i]$, recovering standard lower-triangular causal attention identically.

---

## 2. Experimental Design & Falsification Arms

### Task: Multi-Branch Asynchronous Collaborative Reasoning
- $B=3$ concurrent worker streams:
  - Stream 0 computes target state $v_0 \in \{0, 1, 2, 3, 4\}$.
  - Stream 1 concurrently computes distractor state $v_1 \in \{0, 1, 2, 3, 4\}$.
  - Stream 2 (Synthesizer) receives an asynchronous message from Stream 0 and must output $v_0$.
- **Sanity Check ($B=1$):** Single serial stream. Must achieve $>0.85$ test accuracy (Measured: **1.0000**).
- **Multi-Seed Normal Mode ($B=3$):** Evaluated across seeds `[42, 137]`. Must achieve $>0.90$ test accuracy under concurrent distractor interference (Measured: **1.0000 $\pm$ 0.0000**).
- **Lethal Negative Control 1 (Distractor Channel Leakage):** Deliberately bypasses vector clock concurrency isolation, routing Stream 1's distractor into Stream 2's query. Must collapse to random chance baseline $1/5 = 0.2000$ (Measured: **0.2110**).
- **Lethal Negative Control 2 (Decoupled Causal Payload):** Destroys cross-stream message content at test time, isolating whether accuracy stems from true causal transmission. Must collapse to random chance $0.2000$ (Measured: **0.2100**).

---

## 3. Empirical Results Summary

| Metric | Threshold | Seed 42 | Seed 137 | Mean $\pm$ Std | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sanity Check ($B=1$ Serial)** | $\ge 0.85$ | 1.0000 | 1.0000 | **1.0000 $\pm$ 0.0000** | **PASS** |
| **Multi-Stream ($B=3$ Normal)** | $\ge 0.90$ | 1.0000 | 1.0000 | **1.0000 $\pm$ 0.0000** | **PASS** |
| **Kill Control 1 (Distractor Leak)** | $\le 0.30$ | 0.1980 | 0.2240 | **0.2110** (Chance: 0.2000) | **COLLAPSED** |
| **Kill Control 2 (Decoupled Payload)**| $\le 0.30$ | 0.2020 | 0.2180 | **0.2100** (Chance: 0.2000) | **COLLAPSED** |

---

## 4. Reproduction & Execution
```bash
python projects/c126_vector_clock_attention/run_project.py
```

