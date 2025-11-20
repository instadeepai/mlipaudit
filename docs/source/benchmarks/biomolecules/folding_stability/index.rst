.. _folding_stability_benchmarks:

Folding Stability Benchmark
===========================

Purpose
-------

This benchmark evaluates the ability of a machine-learned interatomic potential
(**MLIP**) to maintain the structural integrity of experimentally determined protein
conformations during molecular dynamics (**MD**) simulations.

Description
-----------

Starting from an experimentally derived X-ray or NMR structure, the benchmark performs an **MD** simulation using the **MLIP**
model in the **NVT** ensemble at **300 K** for **250,000 steps** (250 ps), leveraging the `jax-md <https://github.com/google/jax-md>`_,
as integrated via the `mlip <https://github.com/instadeepai/mlip>`_ library, starting from a solvated structure.

Performance is quantified using the following metrics:

- Retention of the original protein fold, via **RMSD** and **TM-score**.
- Retention of secondary structure elements, via **Secondary Structure matching** (using DSSP).
- Overall compactness, via **Compactness** (radius of gyration analysis).

For more information on each metric, please refer to the following pages:

.. toctree::
   :maxdepth: 1

   RMSD & TM-score <folding_stability>
   Compactness (Radius of gyration) <compactness>
   Secondary Structure matching <secondary_structure>

Dataset
-------
The dataset is composed by a series of protein structures taken from the `PDB <https://www.rcsb.org/>`_ databank.
They have the following IDs:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - .. figure:: ../img/trp.png
          :width: 100%
          :align: center
          :figclass: align-center

          TRP-cage (PDBid: 2JOF)
     - .. figure:: ../img/chignolin.png
          :width: 100%
          :align: center
          :figclass: align-center

          Chignolin (PDBid: 1UAO)
     - .. figure:: ../img/1cq0.png
          :width: 100%
          :align: center
          :figclass: align-center

          Orexin-beta (PDBid: 1CQ0)
