# atten_accel

INT8 attention accelerator in SystemVerilog

It computes causal attention, `O = softmax(Q·Kᵀ)·V`, for sequences of up to 64 tokens on one 8×8 systolic array.

![utilization](docs/tiled_utilization.png)

## How it works

The design uses one 8×8 grid of multiply-accumulate units for both steps of attention: first Q·Kᵀ, then P·V.

- Each unit holds two sets of weights: one in use and one being loaded. A swap signal moves through the grid together with the data, so the next job starts without waiting and the grid stays busy.
- The sequence is processed in blocks of 8 keys. Softmax is computed block by block (online softmax), keeping a running maximum and running sum for each row, so the full score matrix is never stored.
- Division in softmax is replaced by a Newton-Raphson reciprocal.
- For causal attention, blocks of future tokens are skipped completely. The design can also skip old blocks, which allows local (sliding-window) attention.

## Files

| File | Description |
|---|---|
| `rtl/pe_db.sv` | One processing element with two weight registers |
| `rtl/attn_core_db.sv` | The 8×8 array |
| `rtl/osm.sv` | Online softmax |
| `rtl/normalizer.sv` | Final division using Newton-Raphson |
| `rtl/attn_tiled.sv` | Top level: scheduling, weight loading, output accumulation |
| `tb/tb_tiled.sv` | Testbench |
| `model/attn_model.py` | Bit-exact Python model of the hardware |
| `formal/` | Formal verification setup |
| `workload/` | Scripts to train small GPT models and test them on the hardware model |
| `compress/` | Pruning, quantization and distillation experiments |

## Results

| | |
|---|---|
| Clock speed | 349 MHz (Nangate45, post synthesis) |
| Area | 287,770 µm² |
| Array utilization | 74% on 64-token sequences |
| Throughput | 27.6 GOPS effective |
| Cycles for one 64-token attention head | 932 |

Synthesis was done with Yosys and timing was checked with OpenSTA. Place and route was not done.

## Testing

- **Simulation:** 2,900 tests in Verilator. Every output matches the Python model exactly, and is also checked against normal floating-point attention.
- **Formal verification:** using SymbiYosys, it is proven that the scheduler never overwrites weights or data before they are used, and that every job finishes. Four deliberately inserted bugs were all caught.

## Accuracy

Two small GPT models were trained on English text, and every attention layer was run through the hardware model. Over 32,000 test tokens per model, the hardware showed no measurable loss in accuracy compared to standard 8-bit attention.

![accuracy](docs/gpt_gating.png)

## Compression experiments

The `compress/` folder tests pruning, quantization and distillation on the same models, measured in accelerator cycles. For example, limiting attention to the last 24 tokens uses 74% of the cycles with almost no loss in accuracy. Details are in `compress/README.md`.

## How to run

You need Verilator 5 or newer and Python with numpy, matplotlib and torch. Formal verification also needs Yosys, sv2v and SymbiYosys.

```
make regress      # run all simulation tests
make formal       # run formal verification
make gpt          # train the models and test them on the hardware
make compress     # run the compression experiments
make throughput   # compute clock speed, utilization and GOPS
```
