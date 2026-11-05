PY ?= python

.PHONY: help setup smoke core adapt snr tables figures all clean

help:
	@echo "setup    install dependencies"
	@echo "smoke    check that the three corpora load and window"
	@echo "core     main grid (3 datasets x 2 splits x 3 models x 2 norms + 6 cross directions)"
	@echo "adapt    few-shot sweep, k = 5/10/20, three seeds"
	@echo "snr      noise sweep, SNR = 8/4/0/-4 dB"
	@echo "tables   regenerate results/tables.md"
	@echo "figures  regenerate figures/*.pdf and *.png"
	@echo "all      core + adapt + snr + tables + figures"

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) -m src.smoke

core:
	$(PY) -m src.run_experiment --core

adapt:
	$(PY) -m src.run_experiment --sweep adapt

snr:
	$(PY) -m src.run_experiment --sweep snr

tables:
	$(PY) -m src.summarize

figures:
	$(PY) -m src.make_figures

all: core adapt snr tables figures

clean:
	rm -rf src/__pycache__
