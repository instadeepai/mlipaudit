.. _stability:

Stability testing
=================

Purpose
-------

To assess the long-term dynamical stability of a machine-learned interatomic potential (**MLIP**) during realistic,
molecular dynamics (**MD**) simulations.

Description
-----------

For each system in the dataset, the benchmark performs a **MD** simulation using the  **MLIP** model in the
**NVT** ensemble at **300 K** for **100,000 steps** (100 ps), leveraging the
`jax-md <https://github.com/google/jax-md>`_, as integrated via the `mlip <https://github.com/instadeepai/mlip>`_
library. The test monitors the system for signs of instability by detecting abrupt temperature spikes
(explosions) and hydrogen atom drift. These indicators help determine whether the **MLIP** maintains
stable and physically consistent dynamics over simulation times.

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

The stability dataset is composed by a series of protein structures, RNA fragments,
peptides and small-molecules experimental structures taken from the `PDB <https://www.rcsb.org/>`_ databank.
They have the following ids:

* 1JRS (Leupeptin)
* 1UAO (Chignolin)
* 1P79 (RNA Fragment)
* 5KGZ (Protein structure with 634 atoms)
* 1AB7 (Protein structure with 1,432 atoms)
* 1BIP (Protein structure with 1,818 atoms)
* 1A5E (Protein structure with 2,301 atoms)
* 1A7M (Protein structure with 2,803 atoms)

Interpretation
--------------

The **stability score** is a measure of the stability of the **MLIP** model. A score of **1.0** indicates **perfect stability**,
a score of **0.0** indicates **complete instability**.
