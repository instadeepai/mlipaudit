.. _reactivity:

Reactivity
==========

Purpose
-------

This benchmark assesses the **MLIP**'s capability to predict the energy of transition states (TS) and thereby
the activation energy and enthalpy of formation of a reaction. Accurately modeling chemical reactions is an
important use case to employ MLIPs to understand reactivity and to predict the outcomes of chemical reactions.

.. figure:: img/reactivity.png
    :figwidth: 70%
    :align: center

    Chemical reaction example


Description
-----------

This benchmark leverages the `mlip <https://github.com/instadeepai/mlip>`_ library for model inference,
to predict the energy of reactants, products and transition states of a lare dataset of reactions.
From the difference between these states, the activation energy and enthalpy of formation can be calculated. The
performance is quantified using the **MAE** and **RMSE** in activation energy and enthalpy of formation.

Dataset
-------

The dataset used for this benchmark is the **Grambow** \ [#f1]_ dataset which contains
the reactants, products and transition states of 11960 reactions.

Interpretation
--------------

This benchmark tests the accuracy and ability to represent relative energy differences between the states of a reaction.
Both, **MAE** and **RMSE**, should be as low as possible.

References
----------

.. [#f1] C. A. Grambow, L. Pattanaik, W. H. Green, Scientific Data 2020. DOI: https://doi.org/10.1038/s41597-020-0460-4
