.. _folding:

Folding
=======

Purpose
-------

This benchmark evaluates the ability of a machine-learned interatomic potential
(**MLIP**) to preserve the structural integrity of
experimentally determined protein conformations during molecular dynamics (**MD**)
simulations. Specifically, it assesses the retention
of secondary structure elements and overall compactness across a set of known protein
structures.

The benchmark consists of two distinct evaluation protocols:

- **Native structure maintenance**: Starting from the experimentally determined X-ray
structure, the benchmark assesses the **MLIP**'s ability to maintain the native protein
fold during molecular dynamics simulation.
- **De novo folding simulation**: Starting from a linear (unfolded) polypeptide chain,
the benchmark evaluates whether the **MLIP** can drive the protein toward its native
folded conformation.

Description
-----------

The benchmark performs an **MD** simulation using the **MLIP** model in the **NVT**
ensemble at **300 K** for **100,000 steps**,
leveraging the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library. The starting configuration is an
experimentally determined structure.

For each system, the benchmark compares the following metrics against the reference
structure for each trajectory frame:

- **TM-score (Template Modeling score)**
- **RMSD (Root Mean Square Deviation)**

Both metrics are computed via the compute_tm_scores function, which
uses `mdtraj <https://www.mdtraj.org/>`_. to extract
the carbon alpha from the structure and `tmtools <https://pypi.org/project/tmtools/>`_
to compute **RMSD** and **TM-score**.

.. list-table::
   :widths: 33 33 33
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

The **TM-score** \ [#f1]_ and **RMSD** are two distinct metrics used to
evaluate the structural similarity between protein models or conformations.
**RMSD** measures the average distance between the backbone atoms of superimposed
structures; a **lower** **RMSD** indicates greater similarity**, but it is highly
sensitive to local deviations and misaligned regions, making it less reliable
for assessing overall fold similarity, especially when large conformational changes
or flexible regions are present. In contrast, the **TM-score** is designed to be
sensitive to the global topology of the protein,
**ranging from 0 to 1, where 1 indicates a perfect match and scoresabove 0.5 generally suggest a similar fold**;
it is less affected by local errors or misalignments, providing a more robust measure
of overall structural resemblance
and is often preferred for comparing structures with significant differences or for
assessing the quality of protein models.

Dataset
-------

TRP-cage,
Chignolin,
Amyloid-beta,
Hypocretin-2

Interpretation
--------------

The **TM-score** and **RMSD** metrics should be as low as possible.
The **TM-score** should be **close to 1** and the **RMSD** should be **close to 0**.

References
----------

.. [#f1]  Zhang Y, Skolnick J. Scoring function for automated assessment of
          protein structure template quality.
          Proteins. 2004;57(4):702-710. doi:10.1002/prot.20264
