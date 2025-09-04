.. _stability:

Stability testing
=================

Purpose
-------

To assess the long-term dynamical stability of a machine-learned interatomic potential (**MLIP**) during realistic,
production-scale molecular dynamics (**MD**) simulations.

Description
-----------

For each system in the dataset, the benchmark performs a **MD** simulation using the  **MLIP** model in the
**NVT** ensemble at **300 K** for **1,000,000 steps** (1ns), leveraging the
`jax-md <https://github.com/google/jax-md>`_, as integrated via the `mlip <https://github.com/instadeepai/mlip>`_
library. The test monitors the system for signs of instability by detecting abrupt temperature spikes
(**“explosions”**) and hydrogen atom drift. These indicators help determine whether the **MLIP** maintains
stable and physically consistent dynamics over extended simulation times.

Our **stability score** is computed as:

.. math::

   S =
   \begin{cases}
   \tfrac12\,\dfrac{fₑ}{N}, & fₑ < N \quad(\text{explosion})\\[6pt]
   0.5 + \tfrac12\,\dfrac{fₕ}{N}, & fₑ = N, fₕ < N \quad(\text{H loss})\\[6pt]
   1.0, & fₑ = N, fₕ = N \quad(\text{perfect stability})
   \end{cases}

where N is the number of frames in the simulation, fₑ the frame at which the simulation explodes and fₕ,
the frame at which the first H atom detaches.

Dataset
-------

The structures that are tested for stability are a series of protein structures, RNA fragments, peptides and inhibitors taken from the PDB.
They have the following ids:

* 1JRS
* 1UAO
* 1P79
* 5KGZ
* 1AB7
* 1BIP
* 1A5E
* 1A7M
* 2BQV

Interpretation
--------------

The **stability score** is a measure of the stability of the **MLIP** model. A score of **1.0** indicates **perfect stability**,
a score of **0.0** indicates **complete instability**.
