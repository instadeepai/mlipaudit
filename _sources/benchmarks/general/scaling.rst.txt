.. _scaling:

Scaling
=======

Purpose
-------

This benchmark evaluates how the computational cost of machine-learned interatomic potentials (**MLIP**) scales with system size.
By running single **MD** episodes on a series of molecular systems of increasing size, we systematically assess the
relationship between molecular complexity and inference performance. The results provide insight into the efficiency and
scalability of the **MLIP** implementation, helping to identify potential bottlenecks and guide optimization for large-scale
simulations.

Description
-----------

For each system in the dataset, the benchmark performs a **MD** simulation using the  **MLIP** model in the **NVT** ensemble at **300 K**
for **1000 steps** (1 ps), leveraging the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library. During each simulation, a timer tracks the duration of each episode,
and the average episode time (excluding the first episode to ignore the compilation time) is recorded. After all simulations are complete, the benchmark reports
the **average inference time per averagestep as a function of system size**, providing a direct measure of how the **MLIP** implementation's
computational cost grows with increasing molecular complexity. This allows for the identification of scaling bottlenecks and informs
optimization strategies for large-scale simulations.

Dataset
-------

The scaling dataset is composed of a series of protein structures, RNA fragments,
peptides and small-molecules experimental structures taken from the `PDB <https://www.rcsb.org/>`_ databank.
They have the following ids:

* 1JRS
* 1AY3
* 1UAO
* 1P79
* 5KGZ
* 7CI3
* 1AB7
* 1BIP
* 1A5E
* 1A7M
* 2BQV
* 1J7H
* 1VSQ

Interpretation
--------------

This benchmark does not produce a score but can be used to estimate how a model's simulation speed scales with system size.
