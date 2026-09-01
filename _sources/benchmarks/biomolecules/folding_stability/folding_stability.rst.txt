.. _folding_stability:

RMSD and TM-score
=================

Purpose
-------

The **RMSD** and the **TM-score** \ [#f1]_ are two distinct metrics used to
evaluate the structural similarity between protein models or conformations.
**RMSD** measures the average distance between the backbone atoms of superimposed
structures; a **lower** **RMSD** indicates greater **similarity**. The **TM-score**
is instead designed to be sensitive to the global topology of the protein, and is
normalised by the chain length.

Only the **RMSD** contributes to the benchmark score. The **TM-score** normalises distances by
:math:`d_0 = 1.24 (L - 15)^{1/3} - 1.8`, which is undefined below :math:`L = 15`
and is floored at 0.5 Å by TM-align, so it is not perfectly calibrated for chains as short as
the ones in this benchmark (:math:`L = 10` to :math:`36`). Its absolute values should therefore
not be read against the customary "above 0.5 indicates a similar fold" cutoff, which was
established on domain-sized proteins. It remains informative for comparing models against each
other on the same system, and for following how a single trajectory evolves over time.

The results are presented as the average values over the trajectory.
Evolution of the metrics over time is additionally plotted.

Description
-----------

The implementation works as follows:

- The trajectory is loaded as an mdtraj.Trajectory object (see `mdtraj <https://www.mdtraj.org/>`_)
  and the solvent is removed.

- Carbon alpha atoms are extracted from the trajectory and from the experimental
  reference structure.

- The **RMSD** is computed in Å with ``mdtraj.rmsd``, using the known 1:1 atom
  correspondence between the trajectory and the reference under an RMSD-optimal
  (Kabsch) superposition. A sequence aligner is deliberately not used here: both
  structures are the same molecule, so there is no correspondence to infer, and an
  aligner is free to drop badly displaced residues as gaps, which makes the result
  non-monotonic in the actual structural deviation.

- The **TM-score** is computed on the same carbon alpha atoms with the ``tm_align``
  function of `tmtools <https://pypi.org/project/tmtools/>`_.

Interpretation
--------------

A **RMSD** closer to 0 indicates a better match to the reference structure. The score
threshold is 2.0 Å, which is the conventional definition of the folded state for
miniproteins of this size \ [#f2]_.

The **TM-score** ranges from 0 to 1, where 1 indicates a perfect match, subject to the
normalisation caveat described under Purpose above.

References
----------

.. [#f1]  Zhang Y, Skolnick J. Scoring function for automated assessment of
          protein structure template quality.
          Proteins. 2004;57(4):702-710. doi:10.1002/prot.20264

.. [#f2]  Lindorff-Larsen K, Piana S, Dror RO, Shaw DE. How fast-folding
          proteins fold. Science. 2011;334(6055):517-520.
          doi:10.1126/science.1208351
