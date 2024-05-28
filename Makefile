# Get the absolute path of the Makefile
PATH := $(shell cd "$(dirname "$0")" && pwd -P)

SRC_PATH = $(PATH)/src

install:
	@echo "The script is located at: " $(SRC_PATH)
	pip install -e $(SRC_PATH)/mf-prior-bench/
	pip install -e $(SRC_PATH)/PFNs4HPO/
	pip install -e $(SRC_PATH)/neps_lcpfn_hpo/
	pip install -e $(SRC_PATH)/pfns_hpo/
	pip install -r core_requirements.txt
