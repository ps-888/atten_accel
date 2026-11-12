# Compression lab: pruning, quantization and distillation, measured on the accelerator

Model under test: the trained char-level GPT `workload/tiny_gpt.pt`. It has 4 layers, 8 heads, d_model 64, **d_head 8 = accelerator N**, a 64-token context (8 key tiles), 216k parameters, and 196,608 weights in its linear layers.

Every number below comes from the same protocol:

| | |
|---|---|
| fine-tune recipe | 400 steps, batch 32, AdamW **lr 3e-4**, cosine (identical for every model) |
| **control** | the *dense* model fine-tuned with the same recipe. All deltas are paired against it. |
| evaluation | 512 held-out sequences (32,768 tokens), per-token losses, 95% bootstrap CI over sequences |
| hardware cost | accelerator attention cycles per 64-token sequence, from the RTL latency table. Dense = 29,824 cycles (32 heads × 932). |
| hardware check | selected models are re-run with **every surviving head executed by the bit-exact `attn_tiled` model** |

Units: Δ loss in **milli-nats per character** (1 = 0.001 nats/char). Base model 1.77849; dense control 1.77980.

> **Lesson 0 (found by the protocol, not by plan):** fine-tuning the *dense* model at lr 1e-3 made it **7 milli-nats worse**. It knocks the converged model, which ended training at lr≈0, out of its minimum. Without a dense control that loss would have been blamed on whatever compression was being tested. At lr 3e-4 the control is +1.3 (CI 0.0–2.7) vs the base.

## Results

### 1. Unstructured magnitude pruning (global, linear weights): no accelerator benefit

| sparsity | one-shot | one-shot + FT | gradual (cubic) + FT |
|---|---|---|---|
| 30% | +58 | **+12** | +17 |
| 50% | +473 | **+84** | +105 |
| 70% | +1301 | +374 | **+359** |
| 80% | +1744 | +634 | **+588** |
| 90% | +2244 | **+923** | +940 |

* **Fine-tuning recovers 80–90% of the damage.** One-shot numbers are meaningless on their own.
* **Gradual pruning only wins at high sparsity** (70–80%). At 30–50%, a single prune plus fine-tune is better on this budget.
* **This model barely compresses.** At d_model 64 there is little redundancy. Large LLMs tolerate 50% unstructured sparsity almost for free; a 216k-parameter model does not.
* **No accelerator benefit:** the weights run off-chip.

### 2. Structured N:M sparsity

| pattern | sparsity | one-shot | + FT |
|---|---|---|---|
| 2:4 | 50% | +844 | +183 |
| 1:4 | 75% | +1810 | +668 |

2:4 costs **2.2× more than unstructured 50%** (+183 vs +84). That is the price of hardware-friendly structure on a model this small: every row must lose exactly half of every group of 4.

### 3. Head pruning: direct accelerator speedup

Head importance uses the Michel et al. (2019) gradient score, normalized per layer.

| heads removed (of 32) | attention cycles | one-shot | + FT | random heads + FT |
|---|---|---|---|---|
| 4 | 87.5% | +43 | **+10** | |
| 8 | 75.0% | +260 | **+28** | +65 |
| 12 | 62.5% | +328 | **+52** | |
| 16 | 50.0% | +531 | **+92** | +271 |
| 20 | 37.5% | +690 | +134 | |
| 24 | 25.0% | +933 | +200 | |

* **Importance scoring matters:** it costs 2.3× less loss than random selection at −8 heads and 2.9× less at −16.
* **The hardware result matches the FP32 result.** Hardware-in-the-loop vs FP32 for the same pruned model: −8 heads +0.10 [−0.19, 0.39], −16 heads −0.06 [−0.35, 0.25]. Pruned heads are simply never scheduled.

### 4. Block-local attention: hardware co-design

This needed one RTL change: `cfg_tile_lo`, a first-key-tile register in the scheduler. Query block `qb` runs only key tiles `[qb − w + 1, qb]`, and skipped tiles cost zero cycles. The change is verified on 2,900 bit-exact regression vectors with the FP check, a windowed job takes exactly the cycles of an unwindowed job with the same tile count, and the formal suite passes at N=4 with all 4 mutants caught. The model is fine-tuned with the **same tile-granular mask** the hardware executes.

| window | attention cycles | zero-shot | + FT | HW vs FP32 (same model) |
|---|---|---|---|---|
| 8 tokens (1 tile) | 56.7% | +485 | +202 | |
| **16 tokens (2 tiles)** | **62.7%** | +92 | **+10.8 [8.8, 12.6]** | +0.17 [−0.12, 0.45] |
| **24 tokens (3 tiles)** | **74.2%** | +58 | **+5.9 [4.4, 7.2]** | +0.35 [0.06, 0.64] |
| 32 tokens (4 tiles) | 82.8% | +33 | +3.9 [2.8, 5.0] | |

![tradeoff](docs/cycles_tradeoff.png)

**The key co-design result:** for the same cycle budget, a local window costs **about 5× less loss than head pruning**:

| cycle budget | window | head pruning |
|---|---|---|
| ~75% | +5.9 | +28 |
| ~63% | +10.8 | +52 |

This char-level model mostly attends locally, so far tiles are cheap to drop, whereas every head carries unique information. The right compression depends on the workload's structure, and the hardware has to expose the knob; here that is a 3-line scheduler change.

### 5. Quantization (weights; attention is already INT8 on the accelerator)

| format | bits/weight | PTQ | QAT (+FT) |
|---|---|---|---|
| INT8 per-channel | 8 | **−0.8 [−2.2, 0.6]** (free) | |
| INT6 per-channel | 6 | +1.7 [0.0, 3.4] | |
| INT4 per-channel | 4 | +66 | +18 |
| INT4 group-16 | 4 (+scale per 16) | +36 | **+10.3 [8.4, 12.4]** |
| INT3 group-16 | 3 (+scale per 16) | +262 | +47 |

* **INT8 needs nothing.**
* **INT4 needs both** quantization-aware training (about 3–4× less damage than PTQ) and fine-grained group scales (about 2× less than per-channel).

**Attention-path QAT.** The model was fine-tuned with fake INT8 Q/K/V identical to the accelerator's per-head quantization. The hardware-vs-FP32 gap shrank from +0.49 [0.18, 0.80] (control) to +0.31 [−0.00, 0.63]. That is a plausible improvement but **not statistically significant**, since the CIs overlap. The attention path was already close to lossless.

### 6. Knowledge distillation

The student is 2 layers (half depth), with the same d_head 8, so it runs on the same hardware. It has 116k parameters vs 216k and uses 50% of the attention cycles. It was trained from random init for 3,000 steps.

| student | loss | Δ vs control |
|---|---|---|
| trained from scratch | 1.9473 | +167 |
| **distilled** (α=0.7, T=2) | **1.9181** | +138 |
| **KD vs scratch** | | **−29 [−36, −23]** |

**Distillation helps significantly**, but a 2-layer student remains far from the teacher at this budget. The teacher had more training (4,000 steps at batch 64), so the absolute gap overstates what an equal budget would show.

### 7. Stacking: naive vs informed

| plan | attention cycles | weights | Δ loss |
|---|---|---|---|
| naive: −8 heads, window 24, 2:4 | 55.7% | 2:4 FP32 | +231 [221, 241] |
| naive: −12 heads, window 16, 2:4 | 39.2% | 2:4 FP32 | +256 |
| **informed: window 24, INT4/g16 QAT** | **74.2%** | **INT4 (8× smaller)** | **+16.5 [14.0, 19.0]** |
| informed: window 16, INT4/g16 QAT | 62.7% | INT4 | +20.5 [17.8, 23.4] |
| informed: −4 heads, window 24, INT4/g16 QAT | 65.0% | INT4 | +26.4 [23.5, 29.4] |

**Effects are not additive, and the worst component dominates.** The naive stack inherited 2:4's damage. The informed stack uses only what sections 1–6 measured as cheap: 74% of the attention cycles and 4-bit weights for +0.017 nats/char. All informed stacks were re-run on the bit-exact hardware model and agree with FP32 to within 0.4 milli-nats.


## Run it

```
python3 compress/experiments.py control      # first: dense fine-tuned control (+ its HW eval)
python3 compress/experiments.py magnitude    # also: heads nm quant distill window combined
python3 compress/plots.py
```
Each experiment takes 1–6 min on one CPU core (`THREADS=n` to change, `FT_LR` to override). The code is organized for reading:
* `common.py` has the mechanisms: masks, N:M, FakeQuant with STE, KD loss, the window mask, and `hw_eval`.
* `experiments.py` has the protocols, one function per technique.

Good exercises:
* Implement movement pruning or Wanda as alternative criteria and compare them with magnitude.
* Try per-layer head budgets instead of a global ranking.
* Learn the window per layer.
* Distill into a model with fewer heads instead of fewer layers.
