.. _installation:

Installation
============

The *mlipaudit* library and command line tools can be installed via pip:

.. code-block:: bash

    pip install mlipaudit

After installation and activating the respective Python environment, the command line
tools `mlipaudit` and `mlipauditapp` should be available.

However, the command above **only installs the regular CPU version** of JAX.
We recommend that the library is run on GPU.
This requires also installing the necessary versions
of `jaxlib <https://pypi.org/project/jaxlib/>`_ which can also be installed via pip. See
the `installation guide of JAX <https://docs.jax.dev/en/latest/installation.html>`_ for
more information.
At time of release, the following install command is supported:

.. code-block:: bash

    pip install -U "jax[cuda12]"

Also, some benchmarks require `JAX-MD <https://github.com/jax-md/jax-md>`_ as a
dependency. As the newest
version of JAX-MD is not available on PyPI yet, this dependency will not
be shipped with *mlipaudit* automatically and instead must be installed
directly from the GitHub repository, like this:

.. code-block:: bash

    pip install git+https://github.com/jax-md/jax-md.git
