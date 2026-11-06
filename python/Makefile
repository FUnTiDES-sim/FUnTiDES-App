SHELL := /bin/bash
.PHONY: integratedTests unitTests docs


integratedTests:
	@if [ ! -z ${venv} ] ; then \
		source ${venv}/bin/activate; \
	else \
		source ${VIRTUAL_ENV}/bin/activate; \
	fi
	@which python3;
	@cd tests/integratedTests && python3 -m pytest -rA -v   # the MPI tests launch mpirun themselves

unitTests:
	@if [ ! -z ${venv} ] ; then \
		source ${venv}/bin/activate; \
	else \
		source ${VIRTUAL_ENV}/bin/activate; \
	fi
	@which python3;
	@cd tests/unitTests && python3 -m pytest -s -rA -vvv

docs:
	@if [ ! -z ${venv} ] ; then \
		source ${venv}/bin/activate; \
	else \
		source ${VIRTUAL_ENV}/bin/activate; \
	fi 
	@cd docs && python3 -m sphinx -b html source build;
