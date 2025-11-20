.. _sampling:

Protein Sampling
================

Purpose
-------
This benchmark evaluates the quality and accuracy of Machine Learning Interatomic Potentials (**MLIP**) by analyzing the
conformational sampling of amino acids in small proteins during molecular dynamics simulations. Specifically, it computes backbone **Ramachandran angles
(phi/psi)** and **side chain rotamer angles (chi1, chi2, ...)**. The sampled probability distribution of these angles is then compared against
reference data \ [#f1]_ and outliers are detected.

Description
-----------
This benchmark evaluates the conformational sampling of the protein simulations of the folding stability benchmark. The
sampled probability distribution of backbone and side chain dihedrals in these simulations is compared to a reference distribution. The main metrics are
the **RMSD** and the **Hellinger distance** between the sampled and reference distributions. We also compute the **outliers ratio**
of the sampled dihedrals. An outlier is defined as a conformation that is far away from any point of the reference data.

Dataset
-------

See dataset section of the folding stability benchmark.

Interpretation
--------------
The **RMSD** and the **Hellinger distance** are measures of the similarity between the sampled and reference distributions.
The lower the value, the more similar the distributions are. The **outliers ratio** provides a measure of how often the
MLIP samples conformations that do not appear in the reference data. The lower the value, the fewer outliers there are. An
MLIP with an outlier ratio higher than 0.3 should be considered as not sampling the protein conformational space correctly.

References
----------
.. [#f1] Lovell, S.C., Word, J.M., Richardson, J.S. and Richardson, D.C. (2000),The penultimate rotamer library. Proteins, 40: 389-408. https://doi.org/10.1002/1097-0134(20000815)40:3<389::AID-PROT50>3.0.CO;2-2
