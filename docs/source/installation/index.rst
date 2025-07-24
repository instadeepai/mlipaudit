.. _installation:

Installation
============

Quickstart
----------

Pypi index coming soon for open-source release!

From source
-----------

To get started, clone the repository::

    git clone https://github.com/instadeepai/mlipaudit

This repository uses `uv <https://github.com/astral-sh/uv>`_ for dependency management
which you should first install. You can then sync the dependencies manually::

    uv sync --all-groups --no-group gpu

or if you have access to CUDA::

    uv sync --all-groups
