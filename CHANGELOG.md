# Changelog

## Release 0.1.5

- Split the Molecular Liquids benchmarks into four separately scored benchmarks:
  `water_radial_distribution`, `water_density`, `solvent_radial_distribution`, and
  `solvent_density`, all grouped under the `"Molecular Liquids"` category. Each now
  contributes its own score, leaderboard column, and UI page. Density is scored in
  its own right (relative deviation of the equilibrium density from the experimental
  reference) rather than only being displayed on the radial-distribution pages.
- Run the shared NPT simulation only once per system group: the radial-distribution
  and density benchmarks of a group share a `reusable_output_id` and identical
  `ModelOutput` signatures, so the density benchmark reuses the trajectory instead of
  repeating the simulation when both are run together.
- Add a `data_name` attribute (and `data_dir` property) to `Benchmark` so that
  several benchmarks can share the same input data directory and HuggingFace archive;
  the density benchmarks reuse the radial-distribution input data without a separate
  upload.
- Add `Water density` and `Solvent density` UI pages showing the equilibrium-density
  summary statistics and a per-frame density time series against the experimental
  reference.
- Add the `nve_energy_conservation` benchmark, which runs short NVE (constant-energy)
  MD simulations across a set of gas-phase and solvated systems and scores how well a
  model conserves the total energy, quantified by the drift of the total energy over
  the trajectory.

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
