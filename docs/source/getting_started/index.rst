.. _getting_started:

Getting started
===============

Our benchmarks all follow the same interface and are designed to work seamlessly with
`mlip <https://github.com/instadeepai/mlip>`_ models. Benchmarks follow the structure outlined
in :ref:`benchmark`.

Benchmarks can be run by directly interacting with the package:

.. code-block:: python

    from mlip.models import Mace
    from mlip.models.model_io import load_model_from_zip
    from mlipaudit.conformer_selection import ConformerSelectionBenchmark

    # Example: load a mace force field
    force_field = load_model_from_zip(Mace, "path/to/model.zip")

    benchmark = ConformerSelectionBenchmark(force_field=force_field)

    benchmark.run_model()
    result = benchmark.analyze()

We also provide a basic script that allows users to run benchmarks sequentially by providing a zip archive(s) for the model(s)
following the `mlip zip <https://instadeepai.github.io/mlip/user_guide/models.html#load-a-model-from-a-zip-archive>`_ format, as
well as a directory for saving the results. Users can then run:

.. code-block:: shell

    uv run mlipaudit -h

    usage: uv run mlipaudit [-h] -m MODELS [MODELS ...]
            -o OUTPUT [--benchmarks {all,conformer_selection,tautomers}
            [{all,conformer_selection,tautomers} ...]]
            [--fast-dev-run]

    Runs a full benchmark with given models.

    options:
      -h, --help            show this help message and exit
      -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                            paths to the model zip archives
      -o OUTPUT, --output OUTPUT
                            path to the output directory
      --benchmarks {all,conformer_selection,tautomers} [{all,conformer_selection,tautomers} ...]
                            List of benchmarks to run.
      --fast-dev-run        run the benchmarks in fast-dev-run mode
