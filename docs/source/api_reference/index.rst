.. _api_reference:

.. module:: src.mlipaudit

API reference
=============

Base classes and utilities
--------------------------

.. toctree::
    :maxdepth: 2

    benchmark
    io
    run_mode
    scoring
    utils/trajectory_helpers
    utils/inference_and_simulation
    ui

Benchmark implementations
-------------------------

.. toctree::
    :maxdepth: 2

    small_molecules/conformer_selection
    small_molecules/dihedral_scan
    small_molecules/noncovalent_interactions
    small_molecules/tautomers
    small_molecules/ring_planarity
    small_molecules/reference_geometry_stability
    small_molecules/bond_length_distribution
    small_molecules/radial_distribution
    small_molecules/solvent_radial_distribution
    small_molecules/reactivity
    small_molecules/nudged_elastic_band
    biomolecules/folding_stability
    biomolecules/sampling
    general/stability
    general/scaling
