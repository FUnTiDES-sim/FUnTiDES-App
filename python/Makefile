# Run from FUnTiDES-App/python/, after `source <FUnTiDES install>/funtides_env.sh`.
SHELL := /bin/bash
.PHONY: install unitTests integratedTests tests docs

install:
	python3 -m pip install -e ".[test]"

unitTests:
	python3 -m pytest tests/unitTests -rA -v

# The MPI tests launch mpirun themselves
integratedTests:
	python3 -m pytest tests/integratedTests -rA -v

tests: unitTests integratedTests

docs:
	python3 -m sphinx -b html docs/source docs/build
