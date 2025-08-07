.. _folding_stability_benchmarks:

Folding Stability Benchmark
===========================

Purpose
-------

This benchmark evaluates the performance of a machine-learned interatomic potential 
(**MLIP**) in preserving the structural integrity of experimentally determined protein 
conformations during molecular dynamics (**MD**) simulations. 
Starting from an experimentally derived X-ray or NMR structure, the benchmark assesses 
the **MLIP** model ability to maintain the native protein fold throughout the simulation.

Specifically, it evaluates the  maintain of the original protein fold, retention of 
secondary structure elements, and overall compactness across a set of known protein 
structures. This is quantified using various metrics, including  **RMSD**  (Root Mean Square Deviation), 
**TM-score** (Template Modeling score), **Secondary Structure matching** (using DSSP), 
and **Compactness** (radius of gyration analysis).




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

For each system, the benchmark compares the following metrics to the reference structure 
for each trajectory frame. 


.. toctree::
   :maxdepth: 1

   Folding Stability (RMSD & TM-score) <folding_stability>
   Compactness (Radius of gyration) <compactness>
   Secondary Structure matching <secondary_structure>

(For detailed descriptions and implementations of each metric, please refer to the pages linked above)
