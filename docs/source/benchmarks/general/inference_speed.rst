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

The benchmark reuses the ``scaling`` dataset (a size-stratified set of protein chains).
For each system it measures, with warm-up and outlier trimming:

* **Model throughput** — the raw model forward pass (energy + forces), independent of
  the simulation engine. For mlip models this is the pure network forward on a
  pre-built graph (mirroring mlip-jax's ``scripts/time_inference.py``); for external
  ASE calculators it is a forced recomputation on the pre-built atoms (which also
  includes the calculator's neighbour-list construction). Reported as **atoms/s**.
* **MD throughput**, per backend — an end-to-end short **NVT** **MD** simulation at
  **300 K**, timed per episode (discarding the first to ignore compilation), reported
  as **ns/day**. mlip models are run on both the **JAX-MD** and **ASE** backends
  (the latter via ``MLIPForceFieldASECalculator``), while external ASE calculators are
  run on **ASE** only. Because ASE is shared across all models, the ASE numbers give an
  apples-to-apples MD comparison between JAX and external models, while JAX-MD shows
  the mlip best case.

The gap between the metrics reflects simulation overhead (neighbour lists, the
integrator and engine); the JAX-MD-vs-ASE gap for an mlip model isolates the backend
overhead specifically. The GUI lets you switch metrics/backends, plots them on log–log
axes with power-law fits and per-system variance, and shows a per-model summary
including the model's **graph cutoff** (which drives neighbour count and therefore
speed).

Dataset
-------

This benchmark reuses the ``scaling`` dataset — see :ref:`scaling` for details of the
size-stratified protein chains.

Interpretation
--------------

The benchmark produces a score in ``[0, 1]`` based on the per-atom **model forward
time** ``t`` via a Hill function ``1 / (1 + (t / t₀)ᵏ)`` averaged over systems, so
faster models score higher. The forward time (rather than the MD step time) is scored
because it is engine-independent. Because ``t`` is wall-clock time, this score is
hardware-dependent and is only comparable across models run on the same GPU.
