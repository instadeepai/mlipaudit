# Changelog

## Unreleased

- Require `mlip>=0.2.0,<0.3.0` and update the `cuda` extra to use `mlip`'s renamed
  `cuda12` extra (the `cuda` extra was removed in `mlip` 0.2.0).
- Support loading `esen` models (added in `mlip` 0.2.0) by recognizing `esen` in the
  model zip file name.
- Fix model loading against `mlip` 0.2.0: disable stress prediction via the predictor's
  `required_properties` (the `predict_stress` flag was removed from the predictor).
- Fix the ASE-calculator simulation engine against `mlip` 0.2.0 by setting the
  `is_md_simulation`/`is_npt_simulation` attributes now expected by the base engine.

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
