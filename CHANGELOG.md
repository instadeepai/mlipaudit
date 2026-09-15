# Changelog

## Release 0.1.7

- Cap the total-energy drift charts on the `nve_energy_conservation` UI page so
  that a diverging model no longer flattens every other curve onto zero.

## Release 0.1.6

- Remove the TM-score from the `folding_stability` score.
- Set total charges to integers across relevant benchmarks.
- Bug fixes to the `stability` result and the `sampling` UI page, and require
  `streamlit>=1.51.0`.

## Release 0.1.5

- Split the Molecular Liquids benchmarks into four separately scored benchmarks:
  `water_radial_distribution`, `water_density`, `solvent_radial_distribution`, and
  `solvent_density`, all grouped under the `"Molecular Liquids"` category
- Add the `nve_energy_conservation` benchmark, which runs short NVE (constant-energy)
  MD simulations across a set of gas-phase and solvated systems and scores how well a
  model conserves the total energy, quantified by the drift of the total energy over
  the trajectory.
- Replace the `scaling` benchmark with a new `inference_speed` benchmark that measures,
  per system size, both **model throughput** and **MD throughput** on each
  supported backend (JAX-MD and ASE for mlip models, ASE for external).
- Add a `data_name` attribute (and `data_dir` property) to `Benchmark` so that
  several benchmarks can share the same input data directory and HuggingFace archive.
- Align the `sampling` benchmark with `folding_stability`: it now runs the same
  systems through the same minimization + MD protocol and shares its input data via
  `data_name`, keeping the two benchmarks' shared (reused) trajectories consistent.
- Add `Water density` and `Solvent density` UI pages showing the equilibrium-density
  summary statistics and a per-frame density time series against the experimental
  reference.
- Require `mlip>=0.2.3` and adapt `ASESimulationEngineWithCalculator` to its refactor
  of `ASESimulationEngine._init_box` into the `resolve_atoms_cell` helper.
- Replace the orexin-beta system with the villin headpiece (PDB `1UNC`) in the
  `folding_stability` benchmark, and run a JAX-MD FIRE energy minimization before the
  production MD (opt-in via `use_jax_md_minimization` on `run_simulation`). The
  `folding_stability` input data now uses a flat directory layout. These are
  results-affecting changes, so scores are not comparable to previous releases.
- Extract the shared biomolecule systems and simulation protocol into
  `mlipaudit.utils.biomolecules` so `folding_stability` and `sampling` cannot drift.
- Update the `conformer_selection` UI page for the Folmsbee dataset.

## Release 0.1.4

- Add an Apache-2.0 `LICENSE` file and declare the license in `pyproject.toml`.
- Update non-covalent interaction UI.

## Release 0.1.3

- Populate `atoms.info["charge"]` and `atoms.info["spin"]` on every benchmark's
  `ase.Atoms` inputs so MLIP models that read these keys receive the correct
  per-system values. Charges come from each benchmark's input schema or, for
  the directory-of-xyz benchmarks (`folding_stability`, `sampling`, `scaling`,
  `stability`), from a new `STRUCTURE_CHARGES` lookup; spin multiplicity is
  the closed-shell default for all systems.
- Add `DEFAULT_CHARGE` and `DEFAULT_SPIN` constants in `mlipaudit.benchmark`
  to give a single place to override the closed-shell-neutral assumption.
- Add a `dataset` constructor flag on `ConformerSelectionBenchmark` for
  selecting between the `wiggle150` and `folmsbee` reference datasets, and
  refactor `run_model` to run a single batched inference call across all
  conformers.
- Ensure this version is fully compatible with recently released mlip v0.2.0+.

## Release 0.1.2

- Minor updates to improve code readability.
- Updating python images used in ci.
- Fix bug in `ScalingBenchmark.analyze()`.
- Ensure to always load `mlip` `ForceField` models without the `predict_stress` flag
  for compatibility with our simulation engines.
- Bump ci image to `python:3.12-slim`.

## Release 0.1.1

- Fixing bug in the Reactivity UI page with the incorrect data being displayed in
  the summary statistics table.

## Release 0.1.0

- Adding benchmark implementations for 15 benchmarks.
- Adding CLI tool that can (a) run benchmarks and (b) display a GUI for
  visualization of benchmark results.
- Adding code for public leaderboard page only displayed on the
  HuggingFace leaderboard of MLIP models.
- Adding extensive code documentation with tutorials.
- Adding extensive unit test coverage.
