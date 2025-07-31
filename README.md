#  🔬 MLIPAudit:  A library to validate and benchmark MLIP models

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mlipbot/e7c79b17c0a9d47bc826100ef880a16f/raw/pytest-coverage-comment.json)
[![Tests and Linters 🧪](https://github.com/instadeepai/mlipaudit-open/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/mlipaudit-open/actions/workflows/tests_and_linters.yaml)

## 👀 Overview

**MLIPAudit** is a Python library for benchmarking and
validating **Machine Learning Interatomic Potential (MLIP)** models,
in particular those based on the [mlip](https://github.com/instadeepai/mlip) library.
It aims to cover a wide range of use cases and difficulties, providing users with a
comprehensive overview of the performance of their models.

## 🚀 Usage

Note that *mlipaudit* is not yet on PyPI. For now, install locally using *uv*:

```bash
uv sync --all-groups
```

Subsequently, one can run the `src/main.py` benchmarking script:

```bash
uv run mlipaudit -h
```

The `-h` flag prints the help message of the script with the info on how to use it:

```text
usage: uv run mlipaudit [-h] -m MODELS [MODELS ...] -o OUTPUT

Runs a full benchmark with given models.

options:
  -h, --help            show this help message and exit
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        paths to the model zip archives
  -o OUTPUT, --output OUTPUT
                        path to the output directory
```

The zip archives must follow the convention that the model name (one of `mace`,
`visnet`, `nequip`) must be part of the zip file name.

The input data for the benchmarks must be provided in a `./data` directory. If not
provided manually, it will be downloaded from HuggingFace automatically the first time
a benchmark will run.

To run the benchmarking app for visualizing the results, just execute

```bash
uv run streamlit run app.py /path/to/results
```

while making sure that the `app_data` directory is located in the same directory
as you're executing this command from.

## Contributing

### Documentation
When contributing please make sure to update the documentation appropriately. You can run the following
to build a version of the documentation locally to view your changes:
```commandline
uv run sphinx-build -b html docs/source docs/build/html
```
The documentation will be built in the `docs/build/html` directory. You can then open the
`index.html` file in your browser to view the documentation.
