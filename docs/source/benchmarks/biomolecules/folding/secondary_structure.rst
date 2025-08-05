.. _secondary_structure:

Secondary Structure
===================

Purpose
-------

This benchmark provides a rigorous evaluation of the quality and accuracy of
Machine Learning Interatomic Potentials (**MLIPs**) by
computing the secondary structure of a protein during molecular dynamics (**MD**)
simulations. Secondary structure elements, such
as alpha-helices and beta-sheets, are fundamental to a protein's local conformation
and overall fold. By tracking the formation,
stability, and transitions of these secondary structures over time, this benchmark
determines if the **MLIP** accurately maintains
the protein's native secondary structure or captures realistic conformational changes.
Significant deviations in secondary
structure content or stability compared to experimental data or well-established force
fields would indicate deficiencies
in the **MLIP**'s ability to accurately describe interatomic forces and energy
landscapes, thus providing a quantitative measure
of its reliability for simulating protein systems.


Description
-----------

The benchmark performs an **MD** simulation using the **MLIP** model in the **NVT**
ensemble at **300 K** for **100,000 steps**, leveraging
the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library. The secondary structure of
proteins is determined using the **DSSP** (Define Secondary Structure
of Proteins) algorithm \ [#f1]_, as implemented in
the `mdtraj <https://www.mdtraj.org/>`_. Python package. For each frame of the
molecular dynamics trajectory, the atomic coordinates are analyzed
to assign secondary structure elements—such as alpha helices,
beta strands, and coils—to each residue.
The process works as follows:

- The trajectory is loaded as an mdtraj.Trajectory object.

- The function :code:`mdtraj.compute_dssp(traj, simplified=False)` is called, which
computes the secondary structure assignment for each residue in every frame.
We run the same analysis for the reference structure to compute the secondary
structure of the native structure.

- The output is a sequence of secondary structure labels (e.g., "H" for helix,
"E" for strand) for each residue and frame.

For each system, the benchmark compares **MLIP**-predicted secondary structure
content against experimental reference data. Performance
is quantified using the following metrics:

- **Secondary structure content**: The number of residues assigned to secondary
  structure motifs (**helix and strand**) is summed and normalized
  by the sequence length.
- **Matching DSSP**: For each frame, the **DSSP** assignment is compared to the
  reference. A match is counted when both have the same **DSSP** code. This count
  is normalized by the sequence length.

Dataset
-------

*To be added.*

Interpretation
--------------

The secondary structure content should be as **close to the reference as possible**.
The matching **DSSP** should be as **close to 1 as possible**.

References
----------

.. [#f1] Kabsch W, Sander C. Dictionary of protein secondary structure:
         pattern recognition of hydrogen-bonded networks in three-dimensional
         structures. Biopolymers. 1983;22(12):2577-637. doi:10.1002/bip.360221211
