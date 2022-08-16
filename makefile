SHELL:=/bin/bash -o pipefail
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON:=python3


all: venv
	source env/bin/activate; python -m pip install --upgrade pip setuptools
	source env/bin/activate; pip install --use-deprecated=legacy-resolver -e .

venv:
	if [ ! -d $(ROOT_DIR)/env ]; then $(PYTHON) -m venv $(ROOT_DIR)/env; fi

