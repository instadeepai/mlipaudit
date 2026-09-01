.. MLIPAudit documentation master file:

MLIPAudit
=========

Overview
--------

**MLIPAudit** is a Python library and app for benchmarking and
validating **Machine Learning Interatomic Potential (MLIP)** models,
in particular those based on the `mlip <https://github.com/instadeepai/mlip>`_ library.
It aims to cover a wide range of use cases and different levels of complexity,
providing users with a comprehensive overview of the performance of their models.
It also provides the option to benchmark models of any origin
(e.g., also those based on PyTorch) via the ASE calculator interface.

MLIPAudit is a tool that can be installed easily via pip, and run via the command
line. For example,

.. code-block:: bash

    mlipaudit benchmark -m /path/to/visnet.zip /path/to/mace.zip -o /path/to/output

runs the complete benchmark suite for two models, `visnet` and `mace` and
stores the results in JSON files in the `/path/to/output` directory. **The results**
**can contain multiple metrics, however, they will also always include a single score**
**that reflects a model's performance on the benchmark on a scale of 0 to 1.**

To visualize these results, we provide a graphical user interface based on
`streamlit <https://streamlit.io/>`_. Just run,

.. code-block:: bash

    mlipaudit gui /path/to/output

to launch the app (opens a browser window automatically and displays the UI).

.. note::

   This project is under active development.

Getting started
---------------

As a first step, we recommend that you check out our :ref:`installation` page. Second,
we provide a simple tutorial on how running the benchmark suite works and how to
customize it. It is available at :ref:`tutorial_cli`.

We also refer you to :ref:`benchmarks` for more information on each benchmark and
to :ref:`model_scores` for more information on the computation of scores for each model.

As MLIPAudit can also be used as a library, adding new benchmarks or building your
own tools based on our benchmark classes, is easily possible. For a tutorial on this
topic, see :ref:`tutorial_new_benchmark`.

Contents
--------

.. toctree::
    :maxdepth: 1

    Installation <installation/index>
    Tutorial: CLI tools <tutorials/cli/index>
    Tutorial: Adding a new benchmark <tutorials/new_benchmark/index>
    Benchmarks <benchmarks/index>
    Model Scores <scores/index>
    API reference <api_reference/index>
