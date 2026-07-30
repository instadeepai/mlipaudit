.. _inference_speed:

Inference Speed
===============

Purpose
-------

This benchmark measures how fast a machine-learned interatomic potential (**MLIP**) runs
and how that speed scales with system size. It reports two complementary metrics so
models can be compared on performance as well as accuracy.

Description
-----------

The benchmark runs on a size-stratified set of protein chains. For each system it
measures, with warm-up and outlier trimming:

* **Model throughput** — the model forward pass (energy + forces). For mlip models this is the pure network forward on a
  pre-built graph (mirroring mlip-jax's ``scripts/time_inference.py``); for external
  ASE calculators it is a forced recomputation on the pre-built atoms (which also
  includes the calculator's neighbour-list construction). Reported as **atoms/s**.
* **MD throughput**, per backend — an end-to-end short **NVT** **MD** simulation at
  **300 K**, with per-step times derived from the cumulative step count reported at
  each logger call (discarding the initial compilation interval), reported
  as **ns/day**. ``mlip`` models are run on both the **JAX-MD** and **ASE** backends
  (the latter via ``MLIPForceFieldASECalculator``), while external ASE calculators are
  run on **ASE** only. Because ASE is shared across all models, the ASE numbers give an
  apples-to-apples MD comparison between JAX and external models, while JAX-MD shows
  the ``mlip`` best case.

The gap between the metrics and engines may reflect simulation overhead (neighbour lists, the
integrator and engine).

Dataset
-------

The dataset is a size-stratified set of protein chains taken from the
`PDB <https://www.rcsb.org/>`_. Chains were sourced from a PISCES cull list
(non-redundant at 25% sequence identity, resolution ≤ 2.0 Å, no chain breaks)
and a curated small-protein list, screened to charge-neutral sequences at pH 7:

* **2JOF** chain A — Trp-cage TC10b mini-protein (284 atoms)
* **1R0R** chain I — turkey ovomucoid third domain (OMTKY3) (748 atoms)
* **3TXS** chain A — bacteriophage 44RR small terminase gp16 (1513 atoms)
* **4QMD** chain A — human envoplakin plakin-repeat domain (3018 atoms)
* **6U1V** chain A — TcsD acyl-CoA dehydrogenase from FK506 biosynthesis (5964 atoms)

Interpretation
--------------

The benchmark produces a score in ``[0, 1]`` based on the per-atom **model forward
time** ``t`` via a Hill function ``1 / (1 + (t / t₀)ᵏ)`` averaged over systems, so
faster models score higher. The forward time (rather than the MD step time) is scored.
Because ``t`` is wall-clock time, this score is
hardware-dependent and is only comparable across models run on the same GPU.

.. note::

   The scored forward time is **not** measured identically across model families, so
   the score is not a strictly fair comparison between ``mlip`` and external models.
   For ``mlip`` models it times the pure network forward on a pre-built graph, with
   neighbour-list construction excluded; for external ASE calculators it times a full
   recomputation that **includes** the calculator's neighbour-list build. External
   models therefore carry a systematic overhead that ``mlip`` models do not, so
   cross-family score comparisons should be treated with caution — the score is most
   meaningful when comparing models of the same family on identical hardware.
