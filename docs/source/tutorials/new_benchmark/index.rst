.. _tutorial_new_benchmark:

Tutorial: Adding a new benchmark
================================

Code pattern for our benchmarks
-------------------------------

As can be seen in our main benchmarking script `src/mlipaudit/main.py`, the
basic pattern of running our benchmarks in code is the following:

.. code-block:: python

    from mlipaudit.benchmarks import TautomersBenchmark
    from mlipaudit.io import write_benchmark_result_to_disk
    from mlip.models import Mace
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(Mace, "./mace.zip")

    benchmark = TautomersBenchmark(force_field)
    benchmark.run_model()
    result = benchmark.analyze()

    write_benchmark_result_to_disk(
        TautomersBenchmark.name, result, "./results/mace"
    )

After initializing a benchmark class, in this example
:py:class:`TautomersBenchmark <mlipaudit.benchmarks.tautomers.tautomers.TautomersBenchmark>`,
we call `run_model()` to execute all inference calls and simulations required with
the MLIP force field model. The raw output of this is stored inside the class.
Next, we run `analyze()` to produce the final benchmarking results. This function
returns the results class which is always a derived class of
:py:class:`BenchmarkResult <mlipaudit.benchmark.BenchmarkResult>`, in this example
:py:class:`TautomersResult <mlipaudit.benchmarks.tautomers.tautomers.TautomersResult>`. The
function :py:meth:`write_benchmark_result_to_disk <mlipaudit.io.write_benchmark_result_to_disk>`
then writes these results to disk in JSON format.

How to implement a new benchmark
--------------------------------

Overview
^^^^^^^^

A new benchmark class can easily be implemented as a derived class of the abstract
base class :py:class:`Benchmark <mlipaudit.benchmark.Benchmark>`. The attributes and
members to override are:

* `name`: A unique name for the benchmark.
* `category`: A string that represents the category of the benchmark.
  If not overridden, "General" is used. Currently, used exclusively for
  visualization in the GUI.
* `result_class`: A reference to the results class of the benchmark.
  More details below.
* `model_output_class`: A reference to the model output class of the benchmark.
  More details below.
* `required_elements`: A set of element symbols that are required by a model to run
  this benchmark.
* `skip_if_elements_missing`: Boolean that has a default of `True` and hence does
  not need to be overridden. However, if you want your benchmark to still run even if
  a model is missing some required elements, then this should be overridden to be
  `False`. A reason for this would be that parts of the benchmark can still be run
  in this case and the missing elements will be handled on a case-by-case basis inside
  the benchmark's run function.
* `run_model`: This method implements running all inference calls and simulations
  related to the benchmark. This method can take a significant time to execute. As part
  of this, the raw output of the model should be stored in a model output class that
  needs to be implemented and must be derived from the base class
  :py:class:`ModelOutput <mlipaudit.benchmark.ModelOutput>`, which is
  a `pydantic <https://docs.pydantic.dev/latest/>`_ model (works similar to
  dataclasses but with type validation and serialization built in). The model output
  of this type is then assigned to an instance attribute `self.model_output`.
* `analyze`: This method implements the analysis of the raw results and returns
  the benchmark results. This works similarly to the model output, where the results
  are a derived class of
  :py:class:`BenchmarkResult <mlipaudit.benchmark.BenchmarkResult>` (also a pydantic
  model).

Hence, to add a new benchmark, three classes must be implemented, the benchmark, model
output, and results class.

Note that we also recommend that a new benchmarks implements a very minimal version
of itself that is run if `self.run_mode == RunMode.DEV`. For very long-running
benchmarks, we also recommend to implement a version for
`self.run_mode == RunMode.FAST` that may differ
from `self.run_mode == RunMode.STANDARD`, however, for most benchmarks this may
not be necessary.

Minimal example implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of a very minimal new benchmark implementation:

.. code-block:: python

    import functools
    from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

    class NewResult(BenchmarkResult):
        errors: list[float]

    class NewModelOutput(ModelOutput):
        energies: list[float]

    class NewBenchmark(Benchmark):
        name = "new_benchmark"
        category = "New category"
        result_class = NewResult
        model_output_class = NewModelOutput
        required_elements = {"H", "N", "O", "C"}

        def run_model(self) -> None:
            energies = _compute_energies_blackbox(self.force_field, self._data)
            self.model_output = NewModelOutput(energies=energies)

        def analyze(self) -> NewResult:
            score, errors = _analyze_blackbox(self.model_output, self._data)
            return NewResult(score=score, errors=errors)

        @functools.cached_property
        def _data(self) -> dict:
            data_path = self.data_input_dir / self.name / "new_benchmark_data.json"
            return _load_data_blackbox(data_path)


The data loading as a cached property is only recommended if the loaded data
is needed in both the `run_model()` and the `analyze()` functions.

Note that the functions `_compute_energies_blackbox` and `_analyze_blackbox` are
placeholders for the actual implementations.

Another class attribute that can be specified optionally is `reusable_output_id`,
which is `None` by default. It can be used to signal that two benchmarks use the exact
same `run_model()` method and the exact same signature for the model output class.
This ID should be of type tuple with the names of the benchmarks in it, see the
benchmarks `Sampling` and `FoldingStability` for an example of this. See the source code
of the main benchmarking script for how it reuses the model output of one for the other
benchmark without rerunning any simulation or inference.

**Furthermore, you need to add an import for your benchmark to the**
`src/mlipaudit/benchmarks/__init__.py` **file such that the benchmark can be**
**automatically picked up by the CLI tool.**

Data
^^^^

The benchmark base class downloads the input data for a benchmark from
`HuggingFace <https://huggingface.co/datasets/InstaDeepAI/MLIPAudit-data>`_
automatically if it does not yet exist locally. As you can see in the minimal example
above, the benchmark expects the data to be in the directory
`self.data_input_dir / self.name`. Therefore, if you place your data in this
directory before initializing the benchmark, it will not try to download anything from
HuggingFace. This mechanism allows the data to be provided in custom ways.

UI page
^^^^^^^

To create a new benchmark UI page, we refer to the existing implementations located in
`src/mlipaudit/ui` for how to add a new one. The basic idea is that a page is
represented by a function like this:

.. code-block:: python

    def new_benchmark_page(
        data_func: Callable[[], dict[str, NewResult]],
    ) -> None:
        data = data_func()  # data is a dictionary of model names and results

        # add rest of UI page implementation here
        pass

The implementation must be a valid `streamlit <https://streamlit.io/>`_ page.

In order for this page to be automatically included in the UI app, you need to wrap
this new benchmark page in a derived class of
:py:class:`UIPageWrapper <mlipaudit.ui.page_wrapper.UIPageWrapper>` like this,

.. code-block:: python

    class NewBenchmarkPageWrapper(UIPageWrapper):

        @classmethod
        def get_page_func(cls):
            return new_benchmark_page

        @classmethod
        def get_benchmark_class(cls):
            return NewBenchmark

and then make sure to add the import of your new benchmark page to the
`src/mlipaudit/ui/__init__.py` file. This will result in your benchmark's UI page being
automatically picked up and displayed.

How to run the new benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that as you need to modify some existing source code files of *mlipaudit*
to include your new benchmarks, this cannot be achieved purely with the pip installed
library, however, we recommend to clone or fork our repository and run this local
version instead after adding your own benchmarks with minimal code changes, as explained
above.
