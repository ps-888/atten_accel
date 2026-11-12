# atten_accel

An INT8 causal-attention accelerator in SystemVerilog: `O = softmax(Q·Kᵀ·s)·V` over sequences of `MAXT·N` tokens on a single N×N weight-stationary systolic array (default N=8, 64 tokens).

![utilization](docs/tiled_utilization.png)

## Architecture

```
 Q (N×N)  K,V (L×N)                 key tile j = rows jN..jN+N-1
    │        │  loader: Kⱼᵀ / Vⱼ rows → shadow weights (1 row/cycle)
    ▼        ▼
 issue ─► [ N×N array: active + shadow weight per PE, swap token skewed with data ]
  ▲  slots: Q0 Q1 P0 Q2 P1 … Q(T-1) P(T-2) P(T-1)        (tagged results)
  │                     │ S rows                      │ P·V rows
  │                     ▼                             ▼
  │        osm: online softmax (running m, l)   O ← O·α + PⱼVⱼ
  └── p rows (double-buffered by tile parity) ◄──────┘
                                      end: normalizer O/l → O (Q.8)
```

| file | block |
|---|---|
| `rtl/pe_db.sv` | PE with active and shadow weight registers; zero-skip MAC |
| `rtl/attn_core_db.sv` | N×N array, input skew, skewed weight-swap token |
| `rtl/osm.sv` | online softmax: integer-domain running max, LUT exp2, running sum, rescale factor α |
| `rtl/normalizer.sv` | final O/l: leading-one normalization, seed ROM, 2× Newton-Raphson |
| `rtl/attn_tiled.sv` | slot scheduler, weight loader, causal tile skip, tile window (`cfg_tile_lo`), output rescale, perf counters |

* **Zero-bubble streaming.** The swap token travels with the first row of each new stream, so row r switches weights exactly when that stream reaches it. Q·Kⱼᵀ and P·Vⱼ₋₁ streams therefore run back-to-back while the next weight set loads into the shadow registers.
* **Steady state.** Each extra key tile costs exactly 2N cycles. At N=8, latency is 66 → 172 cycles for 1 → 8 tiles, which is 74% array utilization at 64 tokens.
* **Causal skipping.** Future key tiles are never scheduled. `cfg_tile_lo` additionally skips old tiles for block-local attention.

## Verification

| | |
|---|---|
| golden model | `model/attn_model.py`, bit-accurate to every RTL operation |
| regression | `scripts/regress_tiled_all.py`: 2,900 constrained-random and directed vectors over (N, MAXT) = (4,4) (4,8) (8,8) (16,4). Covers causal, tile-window and gating modes, reverse-order loads, mid-run resets and spurious starts. **0 mismatches** on O, the running sums, MAC counters and tile counts |
| independent FP check | every non-gated vector is also compared with exact floating-point attention computed from the same INT8 inputs; worst error 2.8% of full scale |
| formal | `formal/`: SymbiYosys + ABC PDR with the softmax and normalizer abstracted. Unbounded proof at N=4, MAXT=2 of: no shadow-weight overwrite before swap, every swap consumes fresh weights, P rows issued only when ready, no probability row overwritten, everything drained at done, and every job finishes within `(2·MAXT+2)·N+48` cycles. Four injected mutants are all caught |

The FP check found an overflow at sharp softmax temperatures. The golden model mirrored it, so the bit-exact regression could not. The fix keeps the running max on integer scores and scales only the non-negative differences.

## Synthesis (Nangate45, pre-layout)

yosys with liberty mapping, ABC buffering and gate sizing, then OpenSTA. No place-and-route. The reports are in `results/syn/`.

| block | area (µm²) | fmax |
|---|---|---|
| full `attn_tiled`, N=8, 64 tokens | 287,770 | 349 MHz |
| `attn_core_db` (8×8 array) | 82,314 | 387 MHz |

At 349 MHz, one causal 64-token head takes 932 cycles (36 key tiles), which is 79 operations per cycle: **27.6 GOPS effective**, against a 44.6 GOPS peak. `scripts/throughput.py` derives these numbers from the regression latencies and the STA report, and writes them to `results/throughput.json`.

Pipelining raised the full top from 280 to about 350 MHz:
* registered softmax write-back with one-hot write enables
* two-stage output rescale
* quarter-width partial-product multipliers in osm and the normalizer

About a third of the top's area is the Q/K/V/P/O storage in flip-flops, which would be SRAM in a real design.

## Accuracy on trained models

`workload/` trains two small char-level GPTs on NLTK Project Gutenberg text (d_head 8 and 16, 64-token context). It then runs every attention head through the bit-exact hardware model, over 512 held-out sequences (32k tokens) per model:

| | d_head 8 | d_head 16 |
|---|---|---|
| hardware vs INT8 + exact softmax (Δ nats/char, 95% CI) | +0.00004 [−0.00029, 0.00037] | +0.00009 [−0.00017, 0.00035] |
| attention cycles, causal skipping vs none | 932 vs 1,376 | 586 vs 768 |

The hardware's softmax, online rescaling and normalization add no measurable loss beyond INT8 quantization.

![gpt](docs/gpt_gating.png)

## Compression

`compress/` measures pruning, quantization and distillation in accelerator cycles: [compress/README.md](compress/README.md). For example, a 24-token block-local window reaches 74% of the cycles at +0.006 nats/char, about 5× cheaper than head pruning for the same savings.

## Running

Requires Verilator ≥ 5, Python with numpy, matplotlib and torch, and for `make formal` yosys, sv2v and SymbiYosys (`SV2V=path/to/sv2v SBY=path/to/sby.py`). `make gpt` needs the NLTK Gutenberg corpus (`CORPUS=path`).

```
make regress    # 2,900-vector regression + FP check
make formal     # PDR proof + mutants
make gpt        # train the two models and evaluate them on the hardware model
make compress   # compression experiments
make throughput # fmax, utilization and GOPS from the results files
```
