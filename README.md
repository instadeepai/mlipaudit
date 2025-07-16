#  🔬 MLIPAudit:  A library to validate and benchmark MLIP models

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

Subsequently, one can run the `main.py` benchmarking script:

```bash
uv run python main.py -h
```

The `-h` flag prints the help message of the script with the info on how to use it:

```text
usage: python main.py [-h] -m MODELS [MODELS ...] -o OUTPUT

Runs a full benchmark with given models.

options:
  -h, --help            show this help message and exit
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        paths to the model zip archives
  -o OUTPUT, --output OUTPUT
                        path to the output directory
```

The input data for the benchmarks must be provided in a `./data` directory. If not
provided manually, it will be downloaded from HuggingFace automatically the first time
a benchmark will run.
