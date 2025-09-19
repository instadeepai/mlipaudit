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

The **Wiggle150** \ [#f1]_ dataset of highly strained conformers, contains 50 conformers for each of
three representative drug-like molecules: Adenosine, Benzylpenicillin, and Efavirenz.

.. list-table::
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: img/ado00.png
          :width: 100%
          :align: center
          :figclass: align-center

          Adenosine
     - .. figure:: img/bpn00.png
          :width: 100%
          :align: center
          :figclass: align-center

          Benzylpenicillin
     - .. figure:: img/efa00.png
          :width: 100%
          :align: center
          :figclass: align-center

          Efavirenz


Interpretation
--------------

This benchmark assesses the numerical accuracy and the ability to preserve relative conformer energies of the **MLIP**'s
energy inference method. This is critical for downstream applications like conformer sampling and ranking. The **MAE** and
**RMSE** of the energy inference should be **as low as possible** and match the expectations on accuracy of the **MLIP** during training
and testing. Since the energy differences in this dataset are rather large, the **Spearman correlation** should be **close to 1**.

References
----------

.. [#f1] R. Brew, [...], C. Wagen, ChemRxiv 2025. DOI:10.26434/chemrxiv-2025-4mbsk-v3
