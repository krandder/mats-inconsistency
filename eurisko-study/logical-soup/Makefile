# Logical-induction soup harness. Bend2 JS lane via bun (pinned 2.0.29).
BEND_HOME ?= /home/agent/workspaces/eurisko-study/research/bend2-scratch/bend-2.0.29
BEND      := bun $(BEND_HOME)/bend2/main.ts
PY        := PYTHONPATH=$(CURDIR)/.pylib python3
SRC       := $(wildcard src/*.bend)
TESTS     := $(wildcard tests/*_test.bend)
SMOKE_S   ?= 60

.PHONY: check typecheck test pytest smoke resume bench plot clean

check: typecheck test pytest smoke resume
	@echo "make check: GREEN"

typecheck:
	@for f in $(SRC) $(TESTS); do \
	  out=$$($(BEND) $$f --check-only 2>&1); \
	  echo "$$out" | grep -qx 'All terms check.' || { echo "TYPECHECK FAIL $$f"; echo "$$out" | tail -8; exit 1; }; \
	  echo "typecheck ok  $$f"; \
	done

build/%.js: tests/%.bend $(SRC) | build
	@$(BEND) $< -o $@ >/dev/null

build/soup.js: $(SRC) | build
	@$(BEND) src/soup.bend -o $@ >/dev/null

build:
	@mkdir -p build

test: $(patsubst tests/%.bend,build/%.js,$(TESTS))
	@for t in $^; do \
	  out=$$(timeout 300 bun $$t 2>&1); last=$$(echo "$$out" | tail -1); \
	  echo "$$last"; \
	  echo "$$last" | grep -q 'PASS' && ! echo "$$out" | grep -q 'FAIL' || { echo "$$out" | grep FAIL; exit 1; }; \
	done

# scripts/summarize.py and scripts/matrix.sh --dry-run
pytest:
	@out=$$($(PY) -m unittest tests/summarize_test.py tests/gpu_infra_test.py tests/checkpoint_test.py 2>&1) || { echo "$$out"; echo "PYTEST FAIL"; exit 1; }; \
	echo "pytest: summaries, matrix, GPU infra and checkpoints $$(echo "$$out" | grep ^Ran)"

# 256 cells x 10 epochs; must finish under SMOKE_S seconds on the JS lane.
smoke: build/soup.js
	@mkdir -p runs/smoke
	@t0=$$(date +%s.%N); timeout $(SMOKE_S) bun build/soup.js 8 10 1 256 2 > runs/smoke/metrics.csv || { echo "SMOKE FAIL (timeout or error)"; exit 1; }; \
	t1=$$(date +%s.%N); rows=$$(tail -n +2 runs/smoke/metrics.csv | wc -l); \
	printf 'smoke: 256 cells x 10 epochs, %s rows, %.2fs\n' $$rows $$(echo "$$t1-$$t0" | bc); \
	test $$rows -eq 10 || { echo "SMOKE FAIL: expected 10 rows"; exit 1; }
	@echo "cells=256 epochs=10 seed=1 fuel=256 logq=2 w0=1048576 liq=64 market=lmsr questions=periodic+lfsr+and+or" > runs/smoke/config.txt

# chunked run via scripts/run.sh (snapshot + resume) must equal a straight run
RESUME_DIR := build/resume-run
resume: build/soup.js
	@bun build/soup.js 8 10 1 256 4 2097152 0 1048576 64 1 6 3 > build/resume-straight.csv
	@rm -f $(RESUME_DIR)/config.json $(RESUME_DIR)/metrics.csv $(RESUME_DIR)/timing.csv $(RESUME_DIR)/soup.dump
	@scripts/run.sh --seed 1 --K 256 --F 16 --epochs 10 --held-out-family lfsr --release 6 --visible-radius 3 --chunk 3 --out $(RESUME_DIR) >/dev/null
	@cmp -s build/resume-straight.csv $(RESUME_DIR)/metrics.csv || { echo "RESUME FAIL: chunked run differs from straight run"; exit 1; }
	@echo "resume: 4 chunks with snapshots == straight run (10 epochs, held-out release, radius)"

bench: build/vm_bench.js
	@scripts/bench.sh

plot:
	@$(PY) scripts/plot.py runs/$(RUN)/

clean:
	@trash-put build 2>/dev/null || true
