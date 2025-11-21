.. _bond_length_distribution:

Bond Length Distribution
========================

Purpose
-------

This benchmark evaluates the ability of machine-learned interatomic potentials (**MLIP**) to accurately model
the equilibrium bond lengths of small organic molecules during molecular dynamics (MD) simulations. This is an important
test to understand whether the **MLIP** respects basic chemistry throughout simulations. Accurate prediction of bond
length is crucial for capturing the structural and electronic properties of many pharmaceutically and chemically
relevant compounds.

Description
-----------

For each molecule in the dataset, the benchmark performs an **MD** simulation using the **MLIP** model in the **NVT** ensemble at **300 K**
for **1,000,000 steps** (1ns), leveraging the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library, starting from a reference geometry. Throughout the trajectory, the
positions of the bond atoms are tracked, and their deviation from a the reference bond length of the **QM** optimized starting
structure is calculated. The average deviation over the trajectory provides a direct measure of the **MLIP**'s ability to maintain
bond lengths under thermal fluctuations, enabling quantitative comparison to reference data or other models.

.. figure:: img/bond-length.png
    :figwidth: 50%
    :align: center

    Ethane CC bond

Dataset
-------

The dataset is composed of one structure per tested bond type: C-C, C=C, C#C, C-N, C-O, C=O and C-F. The molecular
structures are selected from the **QM9** \ [#f1]_ dataset.

Interpretation
--------------

While fluctuations around the equilibrium bond length are expected during a simulation, the **average bond length should
be very close to the equilibrium bond length from the reference geometry**. The **average deviation should be as low as possible**.

References
----------

.. [#f1] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules,
    Scientific Data 1, 140022, 2014. DOI: https://doi.org/10.1038/sdata.2014.22
