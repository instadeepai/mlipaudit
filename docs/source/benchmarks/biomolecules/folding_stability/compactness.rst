.. _compactness:

Compactness
===========

Purpose
-------

The radius
of gyration (**Rg**) serves as a key metric for assessing the overall structure and
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



This provides a global measure of the protein's spatial extent, defined as the
root-mean-square distance of atomic positions
from the molecule’s center of mass.

.. math::
   R_g = \sqrt{\frac{1}{N} \sum_{i=1}^N \left| \mathbf{r}_i - \mathbf{r}_{\text{COM}} \right|^2}

where :math:`N` is the number of atoms, :math:`\mathbf{r}_i` is the position
vector of atom :math:`i`, and :math:`\mathbf{r}_{\text{COM}}` is the center of mass
of the molecule.

Implementation :


- The radius of gyration is computed for each frame using `mdtraj <https://www.mdtraj.org/>`_ :code:`compute_rg()` function.



Interpretation
--------------

The radius of gyration provides a quantitative measure of protein compactness during simulations.

- **Decreasing radius of gyration**: A decreasing radius of gyration suggests that the **protein is becoming more compact**.

- **Increasing radius of gyration**: An increasing value suggests **expansion or unfolding**.

**Native structure stability**:  In simulations focused on maintaining the native structure,
the **radius of gyration** should remain
**roughly constant around the experimental value**, indicating that the
**MLIP** preserves the protein's native compactness.
