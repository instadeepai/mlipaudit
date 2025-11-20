.. _ring_planarity:

Ring Planarity
==============

Purpose
-------

This benchmark evaluates the ability of machine-learned interatomic potentials (**MLIP**) to preserve the
planarity of aromatic and conjugated rings in small organic molecules during molecular dynamics simulations.
It tests whether the **MLIP** respects the aromaticity throughout the simulations. Accurate modeling of ring planarity is
essential for capturing the structural and electronic properties of many pharmaceutically and chemically relevant compounds.

Description
-----------

For each molecule in the dataset, the benchmark performs an **MD** simulation using the **MLIP** model in the **NVT** ensemble at **300 K**
for **1,000,000 steps** (1ns), leveraging the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library, starting from a reference geometry.
Throughout the trajectory, the positions of the ring atoms are tracked, and their deviation from a perfect plane is quantified
using the root mean square deviation (**RMSD**) from planarity. The ideal plane of the ring is computed using a principal component
analysis of the ring's atoms.The average deviation over the trajectory provides a direct measure of the **MLIP**'s ability to
maintain ring planarity under thermal fluctuations, enabling quantitative comparison to reference data or other models.

.. figure:: img/ring_planarity.png
    :figwidth: 50%
    :align: center

    Benzene OOP bending

Dataset
-------

Starting structures for the simulations were extracted from the **QM9** \ [#f1]_ dataset using SMARTS queries for a small selection of aromatic
ring systems and then selecting the system with the fewest heavy atoms. The selected aromatic systems are: benzene, furan,
imidazole, purine, pyridine and pyrrole.

Interpretation
--------------
Ring planarity should be maintained throughout a simulation if the **MLIP** respects the aromaticity of the systems. For larger
systems, like indole, a slight deviation from the ideal plane is expected, as well as fluctuations due to thermal motion
throughout the simulation. However, the **average RMSD** throughout the simulation should be **small** and **not exceed 0.3 Å**.

References
----------

.. [#f1] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules,
    Scientific Data 1, 140022, 2014. DOI: https://doi.org/10.1038/sdata.2014.22
