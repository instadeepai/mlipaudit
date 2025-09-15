.. _sampling:

Sampling
========

Purpose
-------
This benchmark evaluates the quality and accuracy of Machine Learning Interatomic Potentials (**MLIP**) by analyzing the
conformational sampling of polypeptides during molecular dynamics simulations. Specifically, it computes backbone **Ramachandran angles
(phi/psi)** and **side chain rotamer angles (chi1, chi2, ...)**. The sampled probability distribution of these angles is then compared against
reference data \ [#f1]_ and outliers are detected.

Description
-----------
This benchmark performs an MD simulation using the **MLIP** model in the **NVT** ensemble at **350 K** for **150,000 steps**. The
sampled probability distribution of backbone and side chain dihedrals is compared to a reference distribution. The main metrics are
the **RMSD** and the **Hellinger distance** between the sampled and reference distributions. We also compute the **outliers ratio**
of the sampled dihedrals. An outlier is defined as a conformation that is far away from any point of the reference data.

Dataset
-------
12 tetrapeptides with acetyl and NME termini, in water boxes.

Each sequence was prepared to have a neutral total charge.

Systems were prepared with tleap, using larger boxes to enable proper handeling of long range cutoffs
and minimised and equilibrated with the NPT ensemble with the AMBER force field using openMM.

Boxes of 300 molecules of water were then extracted from the equilibrated systems.

The sequences are as follows:

  - gln_arg_asp_ala
  - trp_phe_gly_ala
  - gly_tyr_ala_val
  - ala_leu_glu_lys
  - met_ser_asn_gly
  - gly_thr_trp_gly
  - ser_ala_cys_pro
  - val_glu_lys_ala
  - pro_met_ile_gln
  - met_val_his_asn
  - glu_gly_ser_arg

Interpretation
--------------
The **RMSD** and the **Hellinger distance** are measures of the similarity between the sampled and reference distributions.
The lower the value, the more similar the distributions are. The **outliers ratio** provides a measure of how often the
MLIP samples conformations that do not appear in the reference data. The lower the value, the fewer outliers there are. An
MLIP with an outlier ratio higher than 0.3 should be considered as not sampling the protein conformational space correctly.

References
----------
.. [#f1] Lovell, S.C., Word, J.M., Richardson, J.S. and Richardson, D.C. (2000),The penultimate rotamer library. Proteins, 40: 389-408. https://doi.org/10.1002/1097-0134(20000815)40:3<389::AID-PROT50>3.0.CO;2-2
