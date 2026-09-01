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
the frame at which the first H atom detaches. We consider a bond to be broken if the H atom's
distance to its bonded atom exceeds 2.5 Angstrom.

Dataset
-------

The stability dataset is composed of a series of small molecule and protein systems. Some systems are solvated, others in vacuum.
The systems are the following:

   - Small molecule (HCNO-only) in vacuum
   - Small molecule containing Sulfur in vacuum
   - Small molecule containing Halogens in vacuum
   - Peptide (Neurotensin) in vacuum
   - Peptide (Oxytocin - contains Sulfur) in vacuum
   - Large protein (1A7M) in vacuum
   - Peptide (Neurotensin) solvated with water and counter-ions
   - Peptide (Oxytocin) solvated with water

The selection ensures that the benchmark systems are representative of the different types of systems that can be encountered in practice.

Interpretation
--------------

The **stability score** is a measure of the stability of the **MLIP** model. A score of **1.0** indicates **perfect stability**,
a score of **0.0** indicates **complete instability** with respect to the benchmark systems.
