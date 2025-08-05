.. _compactness:

Compactness
===========

Purpose
-------

This benchmark aims to evaluate the quality and accuracy of Machine Learning
Interatomic Potentials (**MLIPs**)
by computing the radius of gyration (**Rg**) or compactness of a protein
during molecular dynamics (**MD**) simulations. The radius
of gyration serves as a key metric for assessing the overall structure and
folding behavior of a protein.
By tracking **Rg** over time, this benchmark determines if the MLIP accurately
maintains the protein's native compactness or
captures realistic unfolding/folding transitions, which are fundamental aspects
of protein dynamics. Significant deviations
in **Rg** compared to experimental data or well-established force fields would
indicate deficiencies in the **MLIP's** ability to
accurately describe interatomic forces and energy landscapes, thus providing a
quantitative measure of its reliability for
simulating protein systems.

Description
-----------

The benchmark performs an **MD** simulation using the **MLIP** model in the
**NVT** ensemble at **300 K** for **100,000 steps**,
leveraging the `jax-md <https://github.com/google/jax-md>`_, as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library.

To assess structural compactness, the radius of gyration is computed for each frame
using `mdtraj <https://www.mdtraj.org/>`_ :code:`compute_rg()` function.
This provides a global measure of the protein's spatial extent, defined as the
root-mean-square distance of atomic positions
from the molecule’s center of mass.

.. math::
   R_g = \sqrt{\frac{1}{N} \sum_{i=1}^N \left| \mathbf{r}_i - \mathbf{r}_{\text{COM}} \right|^2}

where :math:`N` is the number of atoms, :math:`\mathbf{r}_i` is the position
vector of atom :math:`i`, and :math:`\mathbf{r}_{\text{COM}}` is the center of mass
of the molecule.

This metric is calculated for each frame of the molecular dynamics trajectory, allowing
for a quantitative comparison of the predicted and reference folding behavior.

Dataset
-------

*to be added*

Interpretation
--------------

- **Rg as a compactness measure**: The radius of gyration provides a
  quantitative measure of protein compactness during folding simulations.

    + **Decreasing radius of gyration**: A decreasing radius of gyration
      indicates that the **protein is becoming more compact**.
    + **Increasing radius of gyration**: An increasing value suggests
      **expansion or unfolding**.

- **Native structure maintenance**: For native structure maintenance simulations,
  the **radius of gyration** should remain
  **relatively stable around the experimental value**, indicating that the
  **MLIP** preserves the protein's native compactness.

- **De novo folding simulations**: For de novo folding simulations,
  the **radius of gyration** should **decrease over time** as the protein
  transitions from an extended conformation toward its native folded state.
