.. _secondary_structure:

Secondary Structure
===================

Purpose
-------

Secondary structure elements, such
as alpha-helices and beta-sheets, are fundamental to a protein's local conformation
and overall fold. By tracking the formation,
stability, and transitions of these secondary structures over time, this benchmark
determines if the **MLIP** accurately maintains
the protein's native secondary structure or captures realistic conformational changes.
Significant deviations in secondary
structure match relative to the native structure provides a quantitative measure
of its reliability for simulating protein systems.

The results are presented as the average values over the trajectory.
Evolution of the metrics over time is additionally plotted.

Description
-----------
The secondary structure of
proteins is determined using the **DSSP** (Define Secondary Structure
of Proteins) algorithm \ [#f1]_, as implemented in
the `mdtraj <https://www.mdtraj.org/>`_ Python package. For each frame of the
molecular dynamics trajectory, the atomic coordinates are analyzed
to assign secondary structure elements—such as alpha helices,
beta strands, and coils—to each residue.

The implementation is as follows:

- The trajectory is loaded as an mdtraj.Trajectory object.

- The function :code:`mdtraj.compute_dssp(traj, simplified=False)` computes the secondary
  structure assignment for each residue and each frame.

- The same analysis is run for the reference structure

- For each frame, the **DSSP** assignment is compared to the reference. A match is counted
  when both have the same **DSSP** code.

Interpretation
--------------

The secondary structure content should be as **close to the reference as possible**.
The matching **DSSP** should be as **close to 1 as possible**.

References
----------

.. [#f1] Kabsch W, Sander C. Dictionary of protein secondary structure:
         pattern recognition of hydrogen-bonded networks in three-dimensional
         structures. Biopolymers. 1983;22(12):2577-637. doi:10.1002/bip.360221211
