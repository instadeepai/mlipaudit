#  🔬 MLIPAudit:  A library to validate and benchmark MLIP models

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.11](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3110/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters](https://github.com/instadeepai/mlipaudit/actions/workflows/tests_and_linters_and_docs_build.yaml/badge.svg?branch=main)](https://github.com/instadeepai/mlipaudit/actions/workflows/tests_and_linters_and_docs_build.yaml)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mlipbot/e7c79b17c0a9d47bc826100ef880a16f/raw/pytest-coverage-comment.json)

## 👀 Overview

**MLIPAudit** is a Python library and app for benchmarking and
validating **Machine Learning Interatomic Potential (MLIP)** models,
in particular those based on the [mlip](https://github.com/instadeepai/mlip) library.
It aims to cover a wide range of use cases and difficulties, providing users with a
comprehensive overview of the performance of their models. It also provides the option
to benchmark models of any origin (e.g., also those based on PyTorch) via the ASE
calculator interface.

## 📦 Installation

MLIPAudit can be installed via pip:

```bash
pip install mlipaudit
```

However, this command **only installs the regular CPU version** of JAX. If benchmarking
native JAX models, we recommend installing the core library along with the GPU
dependencies (`jax[cuda12]` and `jaxlib`) with the following command:
```bash
pip install "mlipaudit[cuda]"
```

## 📖 Documentation

The detailed code documentation that also contains descriptions for each benchmark and
tutorials on how to use MLIPAudit as an applied user,
can be found [here](https://instadeepai.github.io/mlipaudit/).

## 🚀 Usage

MLIPAudit can be used via its CLI tool `mlipaudit`, which can carry out two main tasks:
the benchmarking task and a graphical UI app for visualization of results. Furthermore,
for advanced users that want to add their own benchmarks or create their own app with
our existing benchmark classes, we also offer to use MLIPAudit as a library.

After installation via pip, the `mlipaudit` command is available in your terminal.
Run the following to obtain an overview of two main tasks, `benchmark` and `gui`:

```bash
mlipaudit -h
```

The `-h` flag prints the help message with the info on how to use the tool.
See below, for details on the two available tasks.

### The benchmarking task

The first task is `benchmark`. It executes a benchmark run and can be configured
via some command line arguments. To print the help message for this specific task,
run:

```bash
mlipaudit benchmark -h
```

For example, to launch a full benchmark for a model located at `/path/to/model.zip`,
you can run:

```bash
mlipaudit benchmark -m /path/to/model.zip -o /path/to/output
```

In this case, benchmark results are written to the directory `/path/to/output`. In this
output directory, there will be subdirectories for the benchmarked models, and for the
benchmarks. Each benchmark will contain a `result.json` file with the results.
The results can contain multiple metrics, however, they will also always include a
single score that rates a model's performance on a benchmark on a scale of 0 to 1.

For a tutorial on how to run models that are not native to the
[mlip](https://github.com/instadeepai/mlip) library, see
[this](https://instadeepai.github.io/mlipaudit/tutorials/cli/index.html#providing-external-models)
section of our documentation.

### The graphical user interface

To visualize the detailed results (potentially of multiple models), the `gui` task can
be run. To get more information, run:

```bash
mlipaudit gui -h
```

For example, to display the results stored at `/path/to/output`, execute:

```bash
mlipaudit gui /path/to/output
```

This should automatically open a webpage in your browser with a graphical user interface
that lets you explore the benchmark results visually. This interface was created using
[streamlit](https://streamlit.io/).

**Note**: The zip archives for the models must follow the convention that the model name
(one of `mace`, `visnet`, `nequip` as of *mlip v0.1.3*) must be part of the zip file
name, such that our app knows which model architecture to load the model into. For
example, the aforementioned `model.zip` file name would not work, but instead
`model_mace.zip` or `visnet_model.zip` would be possible.

Benchmarks can also be run on external models, provided either via the ASE calculator
interface or the `ForceField` API for the [mlip](https://github.com/instadeepai/mlip)
library. For more details, see our documentation
[here](https://instadeepai.github.io/mlipaudit/tutorials/cli/index.html#providing-external-models).

### Library

As described in more detail in the
[code documentation](https://instadeepai.github.io/mlipaudit/), the
benchmark classes can also be easily imported into your own Python code base.
Especially, check out the
[API reference](https://instadeepai.github.io/mlipaudit/api_reference/) of our
documentation for details on the available functions.

You can use these functions to build your own benchmarking script and GUI pages for our
app. For inspiration, we recommend to take a look at the main script located
at `src/mlipaudit/main.py` and the implementation of the GUI located at
`src/mlipaudit/app.py`.

## 🤗 Data

The data for the benchmarks is located on HuggingFace
in [this](https://huggingface.co/datasets/InstaDeepAI/MLIPAudit-data) space. The
benchmark classes will automatically download the data into a local `./data` directory
when needed but won't re-download it if it already exists.

## 🏆 Public Leaderboard

A public leaderboard of models can be
found [here](https://huggingface.co/spaces/InstaDeepAI/mlipaudit-leaderboard).
It is based on the same graphical interface as the UI app provided with
this library.

## 🤝 Contributing

To work directly in this repository, run

```bash
uv sync --extra cuda
```

to set up the environment, as this repo uses [uv](https://docs.astral.sh/uv/) for
package and dependency management.

This command installs the main and dev dependency groups. We recommend to check out
the `pyproject.toml` file for more information. Furthermore,
the extra `cuda` installs the GPU-ready version of JAX which is strongly recommended.
If you do not want to install the `cuda` extra (for example, because you are
on MacOS that does not support this standard installation), you can omit the
`--extra cuda` option in the [uv](https://docs.astral.sh/uv/) command.

When adding new benchmarks, make sure that the following key pieces are added
for each one:
* The benchmark implementation (with unit tests)
* The benchmark UI page (add to existing generic unit test for UI pages)
* The benchmark documentation

More information on adding new benchmarks can be found
[here](https://instadeepai.github.io/mlipaudit/tutorials/new_benchmark/)
in our documentation.

To build a version of the code documentation locally to view your changes, you can run:

```commandline
uv run sphinx-build -b html docs/source docs/build/html
```

The documentation will be built in the `docs/build/html` directory.
You can then open the `index.html` file in your browser to view the documentation.

## 🙏 Acknowledgments

We would like to acknowledge beta testers for this library: Marco Carobene,
Massimo Bortone, Jack Sawdon, Olivier Peltre and Alex Laterre.

## 📚 Citing our work

We kindly request that you to cite [our white paper](https://arxiv.org/abs/2511.20487)
when using this library:

L. Wehrhan, L. Walewski, M. Bluntzer, H. Chomet, J.Tilly, C. Brunken and
S. Acosta-Gutiérrez, *MLIPAudit: A benchmarking tool for Machine
Learned Interatomic Potentials*, arXiv, 2025, arXiv:2511.20487.
