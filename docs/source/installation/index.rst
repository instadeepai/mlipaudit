.. _installation:

Installation
============

The *mlipaudit* library and command line tools can be installed via pip:

.. code-block:: bash

    pip install mlipaudit

After installation and activating the respective Python environment, the command line
tools `mlipaudit` and `mlipauditapp` should be available.

However, the command above **only installs the regular CPU version** of JAX.
If benchmarking native JAX models, we recommend installing the core library
along with the GPU dependencies (`jax[cuda12]` and `jaxlib`) with the following command:

.. code-block:: bash

    pip install "mlipaudit[cuda]"
