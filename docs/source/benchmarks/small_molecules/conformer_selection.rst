.. _conformer_selection:

Conformer selection
===================

Purpose
-------

Organic molecules are flexible and able to adopt multiple conformations. These differ in energy due to strain and subtle changes in intramolecular atomic interactions.
This benchmark evaluates the **MLIP**'s ability to identify the most stable conformers within
an ensemble of flexible organic molecules and accurately predict their relative energy
differences. It focuses on capturing subtle intramolecular interactions and strain effects
that influence conformational energies. These metrics assess both numerical accuracy and the **MLIP**'s ability to preserve
relative conformer energetics, which is critical for downstream applications like
conformational sampling and ranking.

Description
-----------

For each system, the benchmark leverages the `mlip <https://github.com/instadeepai/mlip>`_ library for model inference,
comparing the predicted energies and forces against quantum mechanical **QM** reference data. Performance is quantified using
the following metrics:

- **MAE (Mean Absolute Error)** and **RMSE (Root Mean Square Error)** for total energies (in kcal/mol)
- **Spearman rank correlation coefficient** for conformer energy ordering



Dataset
-------

The **Folmsbee** \ [#f1]_ dataset of up to 10 near-minimum conformers of around 700 organic molecules. The reference level of theory for the energy labels is DLPNO-CCSD(T). Below are two examples of conformer ensembles for the molecules "astex_1gkc" and "omegacsd_HEKZAY".

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: img/astex_1gkc.png
          :width: 100%
          :align: center
          :figclass: align-center

     - .. figure:: img/omegacsd_HEKZAY.png
          :width: 100%
          :align: center
          :figclass: align-center


Interpretation
--------------

This benchmark assesses the numerical accuracy and the ability to preserve relative conformer energies of the **MLIP**'s
energy inference method. This is critical for downstream applications like conformer sampling and ranking. The **MAE** and
**RMSE** of the energy inference should be **as low as possible** and match the expectations on accuracy of the **MLIP** during training
and testing. Since the energy differences in this dataset are rather large, the **Spearman correlation** should be **close to 1**.

References
----------

.. [#f1] Dakota Folmsbee, Geoffrey Hutchison; "Assessing conformer energies using electronic structure and machine learning methods" Quantum Chemistry 2020 DOI: https://doi.org/10.1002/qua.26381
