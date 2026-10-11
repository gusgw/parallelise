# Convenience targets. Run "make help" for the list.
PYTHON ?= python3
IMAGE  ?= parallelise:cpu

.PHONY: help install lint test test-fast check-references image

help:  ## Show this list of targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install the package with the development, report and agent extras
	$(PYTHON) -m pip install -e ".[dev,report,agents]"

lint:  ## Check the code style with ruff and the types with mypy
	ruff check .
	$(PYTHON) -m mypy src/parallelise

test:  ## Run every test, including the slow and MPI ones (CUDA tests skip themselves)
	$(PYTHON) -m pytest -q

test-fast:  ## Run the tests that do not compile C code
	$(PYTHON) -m pytest -q -m "not slow"

check-references:  ## Build and test the known-good solution of every problem (skips are listed)
	$(PYTHON) -m pytest -q -rs -m "slow or mpi" -k reference

image:  ## Build the CPU container image
	docker build --target cpu -t $(IMAGE) .
