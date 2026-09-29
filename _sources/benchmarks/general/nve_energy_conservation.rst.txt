.. _nve_energy_conservation:

NVE energy conservation
=======================

Purpose
-------

To assess how well a machine-learned interatomic potential (**MLIP**) conserves the total
mechanical energy during microcanonical (**NVE**) molecular dynamics (**MD**). Energy
conservation is a fundamental physical requirement: under **NVE** dynamics, with no
thermostat exchanging energy with the environment, the total energy
:math:`E(t) = \mathrm{PE}(t) + \mathrm{KE}(t)` should stay constant. A systematic drift
indicates unphysical forces or an inconsistency between the model's energy and its
gradient.

Description
-----------

For each system in the dataset, the benchmark runs a short **NVE** (velocity-Verlet, no
thermostat) **MD** simulation with the **MLIP**, with velocities initialized from a
Maxwell-Boltzmann distribution at **300 K**, leveraging
`jax-md <https://github.com/google/jax-md>`_ as integrated via the
`mlip <https://github.com/instadeepai/mlip>`_ library. The potential and kinetic energies
are recorded at regular snapshots, and the total-energy drift relative to the first frame,
:math:`\Delta E(t) = E(t) - E(0)`, is fitted with a linear model.

The **energy-conservation metric** is the magnitude of the fitted drift over the whole
trajectory, normalized by the kinetic-energy fluctuation scale:

.. math::

   r = \frac{\lvert \text{slope} \rvert \cdot T}{\sigma_{\mathrm{KE}}}

where :math:`T` is the trajectory duration and :math:`\sigma_{\mathrm{KE}}` is the
standard deviation of the kinetic energy along the trajectory. This dimensionless ratio is
mapped to a score in :math:`[0, 1]` via the standard soft threshold: a ratio at or below
**1.0** scores **1.0**, and larger ratios decay exponentially.

The threshold of **1.0** is set at the point where the systematic energy drift accumulated
over the whole trajectory reaches the same magnitude as the natural kinetic-energy
fluctuations: a ratio :math:`r \le 1` keeps the drift within the thermal noise floor, while
:math:`r > 1` means the drift dominates over those fluctuations. This is a deliberately
lenient threshold. Analogous energy-conservation tests for classical force fields typically
use a far tighter ratio, on the order of **0.01**; the more permissive value here reflects
a fundamental difference between the two. A classical force field is an analytic expression
whose potential-energy surface is smooth by construction, whereas an **MLIP** is a highly
parameterized model whose potential-energy surface is learned from finite training data.
While the model is trained to reproduce the training data, it is not guaranteed to be
smooth in between training points, as the interpolation between them can exhibit
high-frequency roughness. This roughness manifests as steep, sudden gradients, which the
MD integrator can struggle with. The resulting numerical error can accumulate and drive
larger integration drift compared to a classical force field. The threshold of **1.0** is
therefore chosen to be lenient enough to accommodate this behavior, while still being
strict enough to flag models that are clearly unphysical.

Dataset
-------

The dataset comprises four representative systems spanning vacuum, bulk liquid and solvated
regimes:

   - Small molecule (HCNO) in vacuum
   - Bulk water (500 molecules)
   - Solvated peptide (Oxytocin)
   - Solvated peptide with counter-ions (Neurotensin)

Systems whose elements the model cannot handle are skipped individually rather than
skipping the whole benchmark.

Interpretation
--------------

A score of **1.0** indicates excellent energy conservation, i.e. the total-energy drift is
small relative to the natural kinetic-energy fluctuations. A score approaching **0.0**
indicates a strongly drifting, non-conservative trajectory.
