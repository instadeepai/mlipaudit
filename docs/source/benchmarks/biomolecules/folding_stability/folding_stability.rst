.. _folding_stability:

TM-score and RMSD
=================

Purpose
-------

The **TM-score** \ [#f1]_ and **RMSD** are two distinct metrics used to
evaluate the structural similarity between protein models or conformations.
**RMSD** measures the average distance between the backbone atoms of superimposed
structures; a **lower** **RMSD** indicates greater **similarity**, but it is highly
sensitive to local deviations and misaligned regions, making it less reliable
for assessing overall fold similarity, especially when large conformational changes
or flexible regions are present. In contrast, the **TM-score** is designed to be
sensitive to the global topology of the protein. It is less affected by local errors
or misalignments, providing a more robust measure of overall structural resemblance.
It is often preferred for comparing structures with significant differences or for
assessing the quality of protein models.

The results are presented as the average values over the trajectory.
Evolution of the metrics over time is additionally plotted.

Description
-----------

The implementation works as follows:

- The trajectory is loaded as an mdtraj.Trajectory object (see `mdtraj <https://www.mdtraj.org/>`_).

- Carbon alpha atoms are extracted from the trajectory

- Both  **RMSD** and **TM-score** are computed on the extracted frames using the compute_tm_scores
  function of `tmtools <https://pypi.org/project/tmtools/>`_

Interpretation
--------------

The **TM-score** ranges from 0 to 1, where 1 indicates a perfect match and scores above 0.5 generally
suggest a similar fold. Anything below 0.5 is considered a poor match.

A **RMSD** closer to 0 indicates a better match to the reference structure.

References
----------

.. [#f1]  Zhang Y, Skolnick J. Scoring function for automated assessment of
          protein structure template quality.
          Proteins. 2004;57(4):702-710. doi:10.1002/prot.20264
