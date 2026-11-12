PY      ?= python3
CORPUS  ?= build/tools/gutenberg
SBY     ?= build/tools/sby/sbysrc/sby.py
.PHONY: luts regress formal gpt compress plots throughput clean

regress: luts
	$(PY) scripts/regress_tiled_all.py
luts:
	$(PY) scripts/gen_luts.py
formal:
	SBY=$(abspath $(SBY)) formal/run_formal.sh
gpt:
	[ -f workload/tiny_gpt.pt ] || $(PY) workload/train_tiny_gpt.py $(CORPUS) workload/tiny_gpt.pt --steps 4000
	[ -f workload/tiny_gpt_hd16.pt ] || $(PY) workload/train_tiny_gpt.py $(CORPUS) workload/tiny_gpt_hd16.pt --steps 4000 --n_head 4
	$(PY) workload/characterize_gpt.py workload/tiny_gpt.pt $(CORPUS) --nseq 512 --tag hd8
	$(PY) workload/characterize_gpt.py workload/tiny_gpt_hd16.pt $(CORPUS) --nseq 512 --tag hd16
compress: gpt
	cd compress && for e in control magnitude heads nm quant distill window combined; do CORPUS=$(abspath $(CORPUS)) $(PY) experiments.py $$e; done && $(PY) plots.py
throughput:
	$(PY) scripts/throughput.py
plots:
	$(PY) scripts/plots.py
clean:
	rm -rf build formal/tiled_* formal/*.log formal/tiled_flat.v
