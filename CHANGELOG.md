# Changelog

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
