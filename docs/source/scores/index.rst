.. _model_scores:

Model Scores
============

To enable consistent and fair comparison across models, we define a composite score
that aggregates performance over all compatible benchmarks. Each
benchmark :math:`b \in \mathcal{B}` may report one or more metrics
:math:`x_{m,b}^{(i)}`, where :math:`i = 1, \ldots, N_b`
indexes the :math:`N_b` metrics evaluated for the model :math:`m`.
For each metric, we compute a normalized score using a soft thresholding
function based on a DFT-derived reference tolerance :math:`t_b^{(i)}` (see Table 1
below):

.. math::

    s_{m,b}^{(i)} =
    \begin{cases}
    1, & \text{if } x_{m,b}^{(i)} \leq t_b^{(i)} \\
    \exp\left(-\alpha \cdot \frac{x_{m,b}^{(i)} - t_b^{(i)}}{t_b^{(i)}}\right), & \text{otherwise}
    \end{cases}

where :math:`\alpha` is a tunable parameter controlling the steepness of the penalty
(e.g., :math:`\alpha = 3`). The per-benchmark score is then computed as the average
over all its metric scores:

.. math::

    s_{m,b} = \frac{1}{N_b} \sum_{i=1}^{N_b} s_{m,b}^{(i)}

Let :math:`\mathcal{B}_m \subseteq \mathcal{B}` denote the subset of benchmarks for
which the model :math:`m` has valid data (i.e., benchmarks compatible with its
element set). The final model score is the mean over all benchmarks on which the
model could be evaluated:

.. math::

    S_m = \frac{1}{|\mathcal{B}_m|} \sum_{b \in \mathcal{B}_m} s_{m,b}

This scoring framework ensures that models are rewarded for meeting or
exceeding DFT-level accuracy.
**In the current version, full benchmarks are skipped if a model does not have all**
**the necessary chemical elements to run all the test cases.**
This is true for all benchmarks, but non-covalent interactions, in which we do a
per-test-case exception.
**When a benchmark is not run,** :math:`s_{m,b} = 0` **is assigned.**
Benchmarks with multiple metrics contribute
proportionally, and the result is a single interpretable score :math:`S_m \in [0,1]`
that balances physical fidelity, chemical coverage, and overall model robustness.
The thresholds for the different benchmarks have been chosen based on the literature.
In the case of tautomers, energy differences are very small; therefore, we've chosen
a stricter threshold of 1-2 kcal/mol, which is not enough for classification.
Thresholds for biomolecules are borrowed from traditional literature in molecular
modeling.



**Table 1: Score thresholds across benchmarks**

.. list-table::
   :header-rows: 1
   :widths: 30 30 20

   * - **Benchmark**
     - **Metric**
     - **Threshold**

   * - Reference Geometry Stability
     - RMSD (Å)
     - 0.075 [#f1]_

   * - Non-covalent Interactions
     - Absolute deviation from reference interaction energy (kcal/mol)
     - 1.0 [#f1]_

   * - Dihedral Scan
     - Mean barrier error (kcal/mol)
     - 1.0 [#f2]_

   * - Conformer Selection
     - MAE (kcal/mol),
       RMSE (kcal/mol)
     - 0.5,
       1.5 [#f3]_

   * - Tautomers
     - Absolute deviation (ΔG)
     - 0.05

   * - Ring Planarity
     - Deviation from plane (Å)
     - 0.05 [#f4]_

   * - Bond Length Distribution
     - Avg. fluctuation (Å)
     - 0.05 [#f1]_

   * - Reactivity-TST
     - Activation Energy (kcal/mol),
       Enthalpy (kcal/mol)
     - 3.0 [#f5]_,
       2.0 [#f5]_

   * - Reactivity-NEB
     - Final force convergence (eV/Å)
     - 0.05 [#f6]_

   * - Radial Distribution Function
     - RMSE (Å)
     - 0.1 [#f7]_

   * - Protein Sampling Outliers
     - Ramachandran ratio,
       Rotamers ratio
     - 0.1,
       0.03

   * - Protein Folding Stability
     - min(RMSD) (Å),
       max(TM-Score)
     - 2.0,
       0.5

References
----------

.. [#f1] \ N. Mardirossian, M. Head‐Gordon, J. Chem. Phys. 2016. DOI: https://doi.org/10.1063/1.4952647
.. [#f2] \ S. Boothroyd, [...], D. L. Mobley, J. Chem. Theory Comput. 2023, 19, 3251–3275. DOI: https://doi.org/10.1021/acs.jctc.3c00039
.. [#f3] \ J. S. Smith, O. Isayev, A. E. Roitberg, Chem. Sci. 2017, 8, 3192–3203. DOI: https://doi.org/10.1039/C6SC05720A
.. [#f4] \ P. R. Evans, Acta Crystallogr. D Biol. Crystallogr. 2007, 63, 58–61. DOI: https://doi.org/10.1107/S090744490604604X
.. [#f5] \ M. Bursch, J.-M. Mewes, A. Hansen, S. Grimme, Angew. Chem. Int. Ed. 2022, 61, e202205735. DOI: https://doi.org/10.1002/anie.202205735
.. [#f6] \ F. Neese et al., ORCA Manual, Section 4.5: Nudged Elastic Band Method, 2024. Available at: https://orca-manual.mpi-muelheim.mpg.de/contents/structurereactivity/neb.html
.. [#f7] \ T. Morawietz, A. Singraber, C. Dellago, J. Behler, Proc. Natl. Acad. Sci. U.S.A. 2016, 113, 8368–8373. DOI: https://doi.org/10.1073/pnas.1602375113
