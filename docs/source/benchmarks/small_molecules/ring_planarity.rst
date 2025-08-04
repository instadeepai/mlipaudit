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

* - .. figure:: img/benzene_oop_bending.png
        :width: 45%
        :align: center
        :figclass: align-center

Benzene OOP bending

Dataset
-------

**to be added**

Interpretation
--------------
Ring planarity should be maintained throughout a simulation if the **MLIP** respects the aromaticity of the systems. For larger
systems, like indole, a slight deviation from the ideal plane is expected and also, fluctuations due to thermal motion
throughout the simulation are expected. However, the **average RMSD** throughout the simulation should be **small** and **not exceed 0.3 Å**.
