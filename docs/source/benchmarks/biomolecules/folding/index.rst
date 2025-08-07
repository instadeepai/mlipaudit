.. _folding_benchmarks:

Folding Stability Benchmark
====================

Purpose
-----------


This benchmark evaluates the ability of a machine-learned interatomic potential
(**MLIP**) to preserve the structural integrity of
experimentally determined protein conformations during molecular dynamics (**MD**)
simulations.  Starting from the experimentally determined X-ray or NMR
structure, the benchmark assesses the **MLIP**'s ability to maintain the native protein
fold during molecular dynamics simulation.


Specifically, it assesses the maintain of original structure folding, the retention
of secondary structure elements and overall compactness across a set of known protein
structures.
This is assessed through different metrics including  **RMSD** (Root Mean Square Deviation),
**TM-score** (Template Modeling score), **Secondary Structure analyses**
(via DSSP), and **Compactness** (radius of gyration analysis).


.. list-table::
   :widths: 25 25 25 25
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
     - .. figure:: ../img/amyloid.png
          :width: 100%
          :align: center
          :figclass: align-center

          Amyloid-beta (PDBid: 1BA6)
     - .. figure:: ../img/1cq0.png
          :width: 100%
          :align: center
          :figclass: align-center

          Hypocretin-2 (PDBid: 1CQ0)


Dataset
-------

TRP-cage (PDBid: 2JOF)

Chignolin  (PDBid: 1UAO)

Amyloid-beta  (PDBid: 1BA6)

Hypocretin-2 (PDBid: 1CQ0)

Description
-----------

The benchmark performs an **MD** simulation using the **MLIP** model in the **NVT**
ensemble at **300 K** for **100,000 steps**,
leveraging the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library. The starting configuration is an
experimentally determined structure (X-ray or NMR).

For each system, the benchmark compares the following  metrics against the reference
structure for each trajectory frame:

For each metric description and implementation please refer to the following pages:

.. toctree::
   :maxdepth: 1

   Folding (RMSD & TM-score) <folding>
   Compactness (Radius of gyration) <compactness>
   Secondary Structure analyses <secondary_structure>
