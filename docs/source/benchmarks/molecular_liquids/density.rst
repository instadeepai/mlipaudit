.. _density:

Density
=======

Purpose
-------

This benchmark assesses the ability of machine-learned interatomic potentials (**MLIP**) to
reproduce the **equilibrium density** of molecular liquids. Density is a fundamental
thermodynamic property: a model that predicts accurate local structure (see the
:ref:`radial_distribution` benchmark) may still get the density badly wrong if the simulation
box expands or collapses. Reproducing the correct density is therefore a complementary and
necessary check on the physical realism of a liquid-phase simulation.

Description
-----------

The benchmark runs the same **MD** simulation as the :ref:`radial_distribution` benchmark: an
**NPT** simulation using the **MLIP** model for **500,000 steps**, leveraging the
`jax-md <https://github.com/google/jax-md>`_ engine from the
`mlip <https://github.com/instadeepai/mlip>`_ library. Water is run at **295.15 K** and **1 atm**,
while all other solvents are run at **293.15 K** and **1 atm**. Because the :ref:`radial_distribution` and density
benchmarks of a system share their input systems and simulation output, the simulation is only
run once when both benchmarks are run together.

The density of each frame is computed from the (fluctuating) simulation cell volume:

.. math::

   \rho = \frac{N_\text{mol} \, M}{N_A \, V}

where :math:`N_\text{mol}` is the number of molecules in the box, :math:`M` is the molecular
weight, :math:`N_A` is Avogadro's number and :math:`V` is the cell volume. The **equilibrium
density** is taken as the average density over the final four fifths of the trajectory (the
first fifth is discarded as equilibration).

Dataset
-------

The benchmark uses the same equilibrated input boxes as the :ref:`radial_distribution` benchmark
(a 500-molecule TIP3P water box, and methanol / acetonitrile / CCl4 boxes built with the GAFF
force field in OpenMM).

Reference densities are the experimental values at the simulation conditions: water
:math:`0.9978\ \text{g/cm}^3`, CCl4 :math:`1.594\ \text{g/cm}^3`, methanol
:math:`0.791\ \text{g/cm}^3` and acetonitrile :math:`0.786\ \text{g/cm}^3`.

Interpretation
--------------

Performance is quantified by the **density deviation**, the absolute difference between the
equilibrium density and the experimental reference. The deviation should be **as low as
possible**. The score is derived from the *relative* deviation (deviation divided by the
reference density) so that it is comparable across liquids of very different densities: a
relative deviation within roughly **2%** scores close to 1, decaying gently beyond that. (The
ideal per-solvent target is ultimately set by the isothermal compressibility of the liquid.) A
large deviation typically indicates that the box has expanded or collapsed during the
simulation, and the density time series can be inspected on the results page to diagnose this.
