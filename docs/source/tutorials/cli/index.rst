.. _tutorial_cli:

Tutorial: CLI tools
===================

After installation and activating the respective Python environment, the command line
tools `mlipaudit`` and ``mlipauditapp`` should be available:

* `mlipaudit``: The **benchmarking CLI tool**. It runs the full or partial benchmark
  suite for one or more models. Results will be stored locally in multiple JSON files
  in an intuitive directory structure.
* `mlipauditapp`: The **UI app** for visualization of the results. Running it opens a
  browser window and displays the web app. Implementation is based
  on `streamlit <https://streamlit.io/>`_.

Benchmarking CLI tool
---------------------

The tool has the following command line options:

* `-h / --help`: Prints info on usage of tool into terminal.
* `-v / --version`: Prints tool version.
* `-m / --models`: Paths to the
  `model zip archives <https://instadeepai.github.io/mlip/user_guide/models.html#load-a-model-from-a-zip-archive>`_.
  If multiple are specified, the tool runs the benchmark suite for all of them
  sequentially. The zip archives for the models must follow the convention that
  the model name (one of `mace`, `visnet`, `nequip` as of *mlip v0.1.3*) must be
  part of the zip file name, such that the app knows which model architecture to load
  the model into. For example, `model_mace_123_abc.zip` is allowed.
* `-o / --output`: Path to an output directory. To this directory, the tool will write
  the results. Inside the directory, there will be subdirectories for each model and
  then subdirectories for each benchmark. Each benchmark directory will hold a
  `result.json` file with the benchmark results.
* `-i / --input`: *Optional* setting for the path to an input data directory.
  If it does not exist, each benchmark will download its data
  from `HuggingFace <https://huggingface.co/datasets/InstaDeepAI/MLIPAudit-data>`_
  automatically. If the data has already been downloaded once, it will not be
  re-downloaded. The default is the local directory `./data`.
* `-b / --benchmarks`: *Optional* setting to specify which benchmarks to run. Can be a
  list of benchmark names (e.g., `dihedral_scan`, `ring_planarity`) or `all` to
  run all available benchmarks. Defaults to `all`. Mutually exclusive with `-e`.
* `-e / --exclude-benchmarks`: *Optional* list of benchmarks to exclude.
  Mutually exclusive with `-b`.
* `-rm / --run-mode`: *Optional* setting that allows to run faster versions of the
  benchmark suite. The default option `standard` which runs the entire suite.
  The option `fast` runs a slightly faster version for some of the very long-running
  benchmarks. The option `dev` runs a very minimal version of each benchmark for
  development and testing purposes.

For example, if you want to run the entire benchmark suite for two models, say
`visnet_1` and `mace_2`, use this command:

.. code-block:: bash

    mlipaudit -m /path/to/visnet_1.zip /path/to/mace_2.zip -o /path/to/output

The output directory then contains an intuitive folder structure of models and
benchmarks with the aforementioned `result.json` files. Each of these files will
contain the results for multiple metrics and possibly multiple test systems in
human-readable format. The JSON schema can be understood by investigating the
corresponding :py:class:`BenchmarkResult <mlipaudit.benchmark.BenchmarkResult>` class
that will be referenced at
the :py:meth:`result_class <mlipaudit.benchmark.Benchmark.result_class>` attribute
for a given benchmark in the :ref:`api_reference`. For example,
:py:class:`ConformerSelectionResult <mlipaudit.benchmarks.conformer_selection.conformer_selection.ConformerSelectionResult>`
will be the result class for the conformer selection benchmark.

Furthermore, each result will also include a that rates a
model's performance on a benchmark on a scale of 0 to 1. For information on what
this score means for a given benchmark, we refer to the :ref:`benchmarks` subsection
of this documentation.

UI app
------

We provide a graphical user interface to visualize the results of the benchmarks located
in the `/path/to/output` (see example above). The app is web-based and can be launched
by running

.. code-block:: bash

    mlipauditapp /path/to/output

in the terminal. This should open a browser window automatically.

The landing page of the app will provide you with some basic information about the app
and with a table of all the evaluated models with their overall score.

On the left sidebar, one can then select each specific benchmark to compare the models
on each one individually. If you have not run a given benchmark, the UI page for that
benchmark will display that data is missing.
