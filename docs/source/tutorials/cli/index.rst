.. _tutorial_cli:

Tutorial: CLI tools
===================

After installation and activating the respective Python environment, the command line
tool `mlipaudit` should be available with two tasks:

* `mlipaudit benchmark`: The **benchmarking CLI task**. It runs the full or partial
  benchmark suite for one or more models. Results will be stored locally in multiple
  JSON files in an intuitive directory structure.
* `mlipaudit gui`: The **UI app** for visualization of the results. Running it opens a
  browser window and displays the web app. Implementation is based
  on `streamlit <https://streamlit.io/>`_.

Benchmarking task
-----------------

The benchmarking CLI task is invoked by running

.. code-block:: bash

    mlipaudit benchmark [OPTIONS]

and has the following command line options:

* `-h / --help`: Prints info on usage of tool into terminal.
* `-m / --models`: Paths to the
  `model zip archives <https://instadeepai.github.io/mlip/user_guide/models.html#load-a-model-from-a-zip-archive>`_
  or to Python files with external model definitions (as either
  `ASE calculator <https://ase-lib.org/ase/calculators/calculators.html>`_ or
  `ForceField <https://instadeepai.github.io/mlip/api_reference/models/force_field.html>`_
  objects). If multiple are specified, the tool runs the benchmark suite for all of them
  sequentially. The zip archives for the models must follow the convention that
  the model name (one of `mace`, `visnet`, `nequip` as of *mlip v0.1.3*) must be
  part of the zip file name, such that the app knows which model architecture to load
  the model into. For example, `model_mace_123_abc.zip` is allowed. For more information
  about providing your own models as ASE calculators or *mlip*-compatible `ForceField`
  classes, see the :ref:`ext_model_tutorial` section.
* `-o / --output`: Path to an output directory. The tool will write
  the results to this directory. Inside the directory, there will be subdirectories for each model and
  then subdirectories for each benchmark. Each benchmark directory will hold a
  `result.json` file with the benchmark results.
* `-i / --input`: *Optional* setting for the path to an input data directory.
  If it does not exist, each benchmark will download its data
  from `HuggingFace <https://huggingface.co/datasets/InstaDeepAI/MLIPAudit-data>`_
  automatically. If the data has already been downloaded once, it will not be
  re-downloaded. The default is the local directory `./data`.
* `-b / --benchmarks`: *Optional* setting to specify which benchmarks to run. Accepts a
  list of benchmark names (e.g., `dihedral_scan`, `ring_planarity`) or `all` to
  run every available benchmark. Default: `all`.  If the flag is omitted, all benchmarks
  run. This is mutually exclusive with `-e`.
* `-e / --exclude`: *Optional* setting to specify which benchmarks to exclude. Works
  in an analogous way to `-b` and is mutually exclusive with it.
* `-rm / --run-mode`: *Optional* setting that allows to run faster versions of the
  benchmark suite. The default option `standard` which runs the entire suite.
  The option `fast` runs a slightly faster version. It runs less test cases for most
  benchmarks and it reduces the number of steps for benchmarks requiring long molecular
  dynamics simulations. The option `dev` runs a very minimal version of each benchmark
  for development and testing purposes. Benchmarks requiring molecular dynamics
  simulations are run with minimal steps.
* `-v / --verbose`: *Optional* flag to enable verbose logging
  from the `mlip <https://github.com/instadeepai/mlip>`_ library code.
* `-lt / --log-timing`: *Optional* flag to enable logging of the run time for each
  benchmark.

For example, if you want to run the entire benchmark suite for two models, say
`visnet_1` and `mace_2`, use this command:

.. code-block:: bash

    mlipaudit benchmark -m /path/to/visnet_1.zip /path/to/mace_2.zip -o /path/to/output

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

Furthermore, each result will also include a score that reflects the
model's performance on the benchmark on a scale of 0 to 1. For information on what
this score means for a given benchmark, we refer to the :ref:`benchmarks` subsection
of this documentation.

UI app
------

We provide a graphical user interface to visualize the results of the benchmarks located
in the `/path/to/output` (see example above). The app is web-based and can be launched
by running

.. code-block:: bash

    mlipaudit gui /path/to/output

in the terminal. This should open a browser window automatically. More information
can be obtained by running `mlipaudit gui -h`.

The landing page of the app will provide you with some basic information about the app
and with a table of all the evaluated models with their overall score.

On the left sidebar, one can then select each specific benchmark to compare the models
on each one individually. If you have not run a given benchmark, the UI page for that
benchmark will display that data is missing.

.. _ext_model_tutorial:

Providing external models
-------------------------

Instead of providing models via `.zip` archives holding models compatible with the
`mlip <https://github.com/instadeepai/mlip>`_ library, we also support any model
to be provided as long as it is implemented as an
`ASE calculator <https://ase-lib.org/ase/calculators/calculators.html>`_ and has
an attribute `allowed_atomic_numbers` of type `set[int]`. Note that the calculator
must have at least the properties `"energy"` and `"forces"` implemented.

The external model also has the choice of following the `ForceField` API of the
`mlip <https://github.com/instadeepai/mlip>`_ library instead (for reference, see
the documentation of this class
`here <https://instadeepai.github.io/mlip/api_reference/models/force_field.html>`_).
This is useful if you have implemented your own MLIP architecture compatible with
the *mlip* library, but not natively included in it. If your model is implemented in
JAX, we strongly recommend to interface it in this way,
because this will allow for making use of highly efficient JAX-MD based simulations
and batched inference in the benchmarks executions. However, if your model is
implemented in PyTorch or another framework, providing it as an ASE calculator is your
best option.

For example, let's assume your model is implemented as an ASE calculator in a module
`my_module` as `MyCalculator`. In this case, you can provide the following code
as a model file `my_model.py`:

.. code-block:: python

    from my_module import MyCalculator

    kwargs = {}  # whatever your configuration is
    mlipaudit_external_model = MyCalculator(**kwargs)

    # Defining that your model can handle H, C, N, and O atoms
    setattr(mlipaudit_external_model, "allowed_atomic_numbers", {1, 6, 7, 8})

Note that in this file, the calculator instance must be initialized and assigned
to a variable that is named `mlipaudit_external_model`.

You can now run your benchmarks like this:

.. code-block:: bash

    mlipaudit benchmark -m /path/to/my_model.py -o /path/to/output

Note that the model name that will be assigned to the model will be `my_model`.

We emphasize that if the object assigned to the variable `mlipaudit_external_model`
is neither of type ASE calculator, nor of type `ForceField` (from the *mlip* API),
a `ValueError` is raised.

If the provided model implementation is based on PyTorch or another deep learning
framework that comes with its own CUDA dependencies, we strongly recommend to not
install the CUDA-based JAX version in the same environment to avoid dependency
conflicts. However, when running the external models, MLIPAudit will not require any
compute-heavy JAX operations, hence, relying on the CPU version of JAX is not an issue
in this case.

.. note::

   MLIPAudit is not optimized for using external models via the ASE calculator
   interface. Hence, it is to be expected that benchmarks can take significantly longer
   compared to using JAX-based and `mlip`-compatible models loaded via
   `.zip` archives.
