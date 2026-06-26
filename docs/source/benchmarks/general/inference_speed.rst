.. _inference_speed:

Inference Speed
===============

Purpose
-------

This benchmark measures how fast a machine-learned interatomic potential (**MLIP**) runs
molecular dynamics, and how that speed scales with system size. It turns raw timings into
an intuitive throughput (ns/day) and a single speed score, so models can be compared on
performance as well as accuracy.

Description
-----------

The benchmark reuses the ``scaling`` dataset (a size-stratified set of protein chains) and
its timing methodology: for each system it runs a short **NVT** **MD** simulation at
**300 K**, timing each episode and discarding the first to ignore compilation. It records
the average step time, the per-episode spread (to quantify variance) and the **MD**
timestep. The GUI reports throughput (ns/day), the fitted scaling exponent
(``step time ∝ Nᵏ``), and a per-model summary.

Dataset
-------

This benchmark reuses the ``scaling`` dataset — see :ref:`scaling` for details of the
size-stratified protein chains.

Interpretation
--------------

The benchmark produces a score in ``[0, 1]`` based on the per-atom step time ``t`` via a
Hill function ``1 / (1 + (t / t₀)ᵏ)`` averaged over systems, so faster models score higher.
Because ``t`` is wall-clock time, this score is hardware-dependent and is only comparable
across models run on the same GPU.
