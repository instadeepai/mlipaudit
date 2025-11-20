.. _nudged_elastic_band:

Nudged Elastic Band
===================

.. note::

    Because the Nudged Elastic Band benchmark is still in its beta phase, we are not
    including it in the public HuggingFace leaderboard at this time. It is, however,
    fully accessible for users running the benchmarks and GUI locally.

Purpose
-------

The nudged elastic band (NEB) is a method to relax a mean energy path between
a reactant and a product structure and thereby find a good guess for the
transition state of the reaction between these two structures. This benchmark assesses
the **MLIP**'s ability to converge NEB calculations where the transition state is already known,
meaning it is a stability benchmark tailored to the NEB method.


Description
-----------

This benchmark uses a custom simulation engine, based on the `ASESimulationEngine` from the `mlip <https://github.com/instadeepai/mlip>`_ library
to run NEB calculations. Before running the NEB calculations, the structures of reactants and products are energy minimized using
the **MLIP** and the **BFGS** optimizer with `alpha=70` and `maxstep=0.03`.
Subsequently, an initial guess for the mean energy path is constructed using the Image  Dependent Pair Potential (IDPP),
placing the known transition state structure in the middle with 10 images between the reactant and product structures.
The path is then relaxed using two NEB runs. The first run is a standard NEB calculation with a force convergence threshold of 0.5 eV/Å.
The second run is a NEB calculation with the climbing image method, with a force convergence threshold of 0.05 eV/Å. Both NEB calculations are run for a maximum of 500 steps.
The technical specifications are chosen to resemble those used in the generation of the **Transition1X** \ [#f2]_ dataset.

Dataset
-------

The dataset used for this benchmark is are 100 reactions sampled from the **Grambow** \ [#f1]_ dataset which contains
the reactants, products and transition states of 11960 reactions.

Interpretation
--------------

This benchmarks tests the ability of the model to converge the NEB calculations. The higher the convergence rate, the better.

References
----------

.. [#f1] C. A. Grambow, L. Pattanaik, W. H. Green, Scientific Data 2020. DOI: https://doi.org/10.1038/s41597-020-0460-4
.. [#f2] M. Schreiner, A. Bhowmik, T. Vegge, J. Busk, Ole Winther, Scientific Data 2022. DOI: https://doi.org/10.1038/s41597-022-01870-w.
