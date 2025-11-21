.. _tautomers:

Tautomers
=========

Purpose
-------

This benchmark assesses the ability of machine-learned interatomic
potentials (**MLIP**) to accurately predict the relative energies and
stabilities of tautomeric forms of small molecules in vacuum.
Tautomers are structural isomers that interconvert via proton transfer
and/or double bond rearrangement,
and accurately estimating the energy gap between them is an important measure
of chemical accuracy in the **MLIP** framework.

.. figure:: img/tautomers.png
    :figwidth: 50%
    :align: center

    Visual representation of the energy difference of a tautomer pair.

Description
-----------

For each molecule, the benchmark leverages the `mlip <https://github.com/instadeepai/mlip>`_ library for model inference,
comparing **MLIP**-predicted energies against quantum mechanical **QM** reference data. Performance
is quantified using the following metrics:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**


Dataset
-------

The benchmark utilizes a dataset of 1,391 tautomer pairs sourced from the
**Tautobase dataset** \ [#f1]_. After generation of the structures
and minimisation at **xtb** level, the **QM** energies were computed
in-house using **ωB97M-D3(BJ)/def2-TZVPPD** level of theory.

Interpretation
--------------

The accuracy of tautomer energy predictions is assessed
through **MAE** and **RMSE** metrics, which should ideally be minimal.
Performance varies considerably across different tautomer classes and
molecular scaffolds. To identify specific weaknesses in the **MLIP**,
examine error patterns by tautomer type and molecular complexity. For problematic
cases, detailed analysis of individual tautomer pairs reveals whether the model
correctly predicts the dominant form and captures the energy differences
between conformations.

References
----------

.. [#f1] Wahl, O., Sander, T., Tautobase: An Open Tautomer
         Database, Journal of Chemical Information and Modeling 2020 60 (3),
         1085-1089, DOI: 10.1021/acs.jcim.0c00035
