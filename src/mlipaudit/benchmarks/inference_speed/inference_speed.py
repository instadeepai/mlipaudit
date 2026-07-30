# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-speed benchmark.

Measures how fast a model runs molecular dynamics and how that speed scales with
system size, and turns it into a single throughput score. It runs on a size-stratified
protein dataset, times the model forward pass and the MD step per backend, records the
per-episode timing spread and the MD timestep, and produces a hardware-relative speed
score.
"""

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from mlipaudit.benchmark import (
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    Benchmark,
    BenchmarkResult,
    ModelOutput,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_speed_score
from mlipaudit.utils.simulation import get_simulation_engine

#: MD simulation configuration: total number of steps, snapshot interval, logger
#: interval, number of episodes and the integration timestep (fs). ``log_interval`` is
#: pinned explicitly (rather than left to mlip's ``ASESimulationConfig`` default)
#: because the ASE backend derives its per-step timing from the cumulative
#: ``state.step`` reported at each logger call: with the default the ASE engine would
#: log only at step 0 and the final step, so every timing chunk (bar the dropped
#: compilation one) would vanish and the ASE MD metric would silently be ``None``. It is
#: set equal to ``snapshot_interval`` so the two sample at a comparable cadence.
#: ``num_episodes`` drives the JAX-MD logging cadence (the ASE backend ignores it). The
#: DEV variant runs a tiny simulation so tests stay fast.
SIMULATION_CONFIG = {
    "num_steps": 1000,
    "snapshot_interval": 100,
    "log_interval": 100,
    "num_episodes": 5,
    "timestep_fs": 1,
}
SIMULATION_CONFIG_DEV = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "log_interval": 1,
    "num_episodes": 10,
    "timestep_fs": 1,
}

#: Number of (smallest) systems to run in DEV mode.
NUM_DEV_SYSTEMS = 2

#: Number of forward passes to discard (JIT compilation / lazy init) and to time when
#: measuring model throughput. After timing, the slowest `FORWARD_TRIM_FRACTION` of
#: passes are dropped (garbage-collection / scheduling spikes) before averaging.
NUM_FORWARD_WARMUP = 2
NUM_FORWARD_TIMED = 25
NUM_FORWARD_TIMED_DEV = 3
FORWARD_TRIM_FRACTION = 0.2

#: MD backend identifiers. ``JAX_MD_BACKEND`` uses mlip's native JAX-MD engine (mlip
#: models only); ``ASE_BACKEND`` uses the ASE engine and is the common backend across
#: all models (mlip models run under ASE via ``MLIPForceFieldASECalculator``), so it
#: gives an apples-to-apples MD comparison between JAX and external models.
JAX_MD_BACKEND = "jax_md"
ASE_BACKEND = "ase"

#: Edge-capacity multiplier used when wrapping an mlip ForceField as an ASE calculator.
EDGE_CAPACITY_MULTIPLIER = 1.25

#: Score parameters for the Hill function ``score = 1 / (1 + (t / t0) ** k)``, where
#: ``t`` is the per-atom model forward-pass time in seconds (the scored, engine-
#: independent metric). ``SCORE_PER_ATOM_FORWARD_TIME_MIDPOINT`` (``t0``) is the
#: per-atom forward time that scores 0.5 and ``SCORE_SHARPNESS`` (``k``) controls how
#: sharply models separate. These are calibrated for our H100 reference hardware; the
#: score is only comparable across models run on the same hardware.
#: TODO: tune ``t0`` against real model runs once available.
SCORE_PER_ATOM_FORWARD_TIME_MIDPOINT = 1.0e-6
SCORE_SHARPNESS = 1.0

logger = logging.getLogger("mlipaudit")


def get_molecule_size_from_name(name: str) -> int:
    """Get the molecule size from the structure name.

    Args:
        name: The name of the structure. Filenames are prefixed with the atom count,
            e.g. ``42_1abc``.

    Returns:
        The number of atoms in the structure.
    """
    return int(name.split("_", maxsplit=1)[0])


class InferenceSpeedModelOutput(ModelOutput):
    """Model output for the inference-speed benchmark.

    Attributes:
        structure_names: The names of the structures used.
        forward_times: A list, per structure, of the individual timed model
            forward-pass durations (excluding warm-up). Empty for structures whose
            forward pass failed.
        md_step_times: A list, per structure, of a mapping from MD backend name
            (``ase``/``jax_md``) to the per-chunk step times (seconds per step) measured
            between successive logger calls, with the compilation chunk excluded.
            Backends that failed or were not run are absent.
    """

    structure_names: list[str]
    forward_times: list[list[float]] = []
    md_step_times: list[dict[str, list[float]]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MDBackendResult(BaseModel):
    """MD throughput for a single backend.

    Attributes:
        average_step_time: The mean MD step time (s), excluding compilation. None if
            the run failed.
        step_time_samples: Per-chunk step times (s/step) between successive logger
            calls (compilation excluded), used for variance. Empty if unavailable.
    """

    average_step_time: NonNegativeFloat | None = None
    step_time_samples: list[float] = []


class InferenceSpeedStructureResult(BaseModel):
    """Result object for a single structure.

    Attributes:
        structure_name: The structure name.
        num_atoms: The number of atoms in the structure.
        num_steps: The number of steps in the simulation.
        num_episodes: The number of episodes in the simulation.
        timestep_fs: The MD timestep in femtoseconds, used to convert step times into
            a throughput (ns/day).
        average_forward_time: The average wall-clock time of a single model forward
            pass (energy + forces), excluding warm-up. This is the engine-independent
            model-throughput metric. None if the forward pass failed.
        forward_times: The individual timed forward-pass durations, used to quantify
            variance. Empty if unavailable.
        md: MD throughput per backend, keyed by backend name (``ase``/``jax_md``).
        failed: Whether the forward pass and all MD backends failed for this structure.
    """

    structure_name: str
    num_atoms: PositiveInt
    num_steps: PositiveInt
    num_episodes: PositiveInt
    timestep_fs: float | None = None
    average_forward_time: NonNegativeFloat | None = None
    forward_times: list[float] = []
    md: dict[str, MDBackendResult] = {}

    failed: bool = False


class InferenceSpeedResult(BenchmarkResult):
    """Result object for the inference-speed benchmark.

    Attributes:
        structure_names: The names of the structures.
        structures: List of per structure results.
        graph_cutoff_angstrom: The model's graph (neighbour-list) cutoff in Angstrom,
            which influences neighbour count and therefore speed. None if unavailable
            (e.g. some external calculators do not expose it).
    """

    structure_names: list[str]
    structures: list[InferenceSpeedStructureResult]
    graph_cutoff_angstrom: float | None = None


class InferenceSpeedBenchmark(Benchmark):
    """Benchmark measuring model and MD throughput and how they scale with size.

    For each structure in the size-stratified protein dataset it measures two
    complementary speeds: the **model forward-pass** time (energy + forces,
    engine-independent) and
    the **MD step** time (end-to-end, including neighbour lists, the integrator and the
    simulation engine). The gap between them reflects simulation overhead. The model is
    scored with a Hill function on its per-atom forward-pass time so that faster models
    score higher; the score is wall-clock based and only comparable across models run on
    the same hardware.

    Attributes:
        name: The unique benchmark name (``inference_speed``), which also determines the
            input data directory and the HuggingFace dataset archive name.
        category: The benchmark category, used for grouping in the UI.
        result_class: The `InferenceSpeedResult` type returned by `analyze`.
        model_output_class: The `InferenceSpeedModelOutput` type.
        required_elements: The element types present in the input files.
    """

    name = "inference_speed"
    category = "General"
    result_class = InferenceSpeedResult
    model_output_class = InferenceSpeedModelOutput

    required_elements = {"N", "H", "O", "S", "C"}

    def run_model(self) -> None:
        """For each structure, time the model forward pass (model throughput) and run a
        short MD simulation on each supported backend (MD throughput). Every
        measurement fails independently.
        """
        forward_times: list[list[float]] = []
        md_step_times: list[dict[str, list[float]]] = []
        for structure_name in self._structure_names:
            try:
                atoms = ase_read(self.data_dir / f"{structure_name}.xyz")
                atoms.info["charge"] = DEFAULT_CHARGE
                atoms.info["spin"] = DEFAULT_SPIN
            except Exception as e:
                logger.info("Error reading system %s: %s", structure_name, str(e))
                forward_times.append([])
                md_step_times.append({})
                continue

            # Model throughput: timed forward passes on a copy to avoid side effects.
            forward_times.append(self._measure_model_throughput(atoms.copy()))

            # MD throughput: a short MD run per supported backend.
            per_backend = {}
            for backend in self._md_backends:
                per_backend[backend] = self._time_md(atoms.copy(), backend)
            md_step_times.append(per_backend)

        self.model_output = InferenceSpeedModelOutput(
            structure_names=self._structure_names,
            forward_times=forward_times,
            md_step_times=md_step_times,
        )

    def _time_md(self, atoms: Any, backend: str) -> list[float]:
        """Run one short MD simulation and return per-chunk step times (s/step).

        For the ASE backend an mlip ``ForceField`` is wrapped in an
        ``MLIPForceFieldASECalculator`` so it runs under the ASE engine (external ASE
        calculators are used directly); for the JAX-MD backend the ForceField is passed
        straight to the JAX-MD engine.

        Timing is derived from the cumulative ``state.step`` at each logger call rather
        than assuming an episode structure: JAX-MD calls loggers per episode, ASE every
        ``log_interval`` steps, so we divide each inter-call wall time by the actual
        number of steps elapsed and drop the compilation chunk (see
        ``_step_times_from_samples``).

        Args:
            atoms: The structure to simulate.
            backend: One of ``ASE_BACKEND`` / ``JAX_MD_BACKEND``.

        Returns:
            The per-chunk step times (s/step), or an empty list on failure.
        """
        from mlip.models import ForceField  # noqa: PLC0415

        try:
            if backend == ASE_BACKEND and isinstance(self.force_field, ForceField):
                from mlip.simulation.ase.mlip_ase_calculator import (  # noqa: PLC0415
                    MLIPForceFieldASECalculator,
                )

                force_field: Any = MLIPForceFieldASECalculator(
                    atoms, EDGE_CAPACITY_MULTIPLIER, self.force_field
                )
            else:
                force_field = self.force_field

            samples: list[tuple[int, float]] = []

            def log(state: Any) -> None:
                samples.append((int(state.step), time.perf_counter()))

            engine = get_simulation_engine(
                atoms=atoms, force_field=force_field, **self._md_kwargs
            )
            engine.attach_logger(log)
            engine.run()
            return self._step_times_from_samples(samples)

        except Exception as e:
            logger.info(
                "Error running %s MD on system %s: %s", backend, str(atoms), str(e)
            )
            return []

    @staticmethod
    def _step_times_from_samples(samples: list[tuple[int, float]]) -> list[float]:
        """Convert ``(cumulative_step, wall_time)`` samples to per-chunk step times.

        Each chunk's step time is its wall-time delta divided by its step-count delta,
        so it is correct regardless of how many steps a backend runs between logger
        calls. The compilation chunk is excluded: if the first sample is at step 0 (the
        ASE case) that opening interval contains compilation and is dropped; for JAX-MD
        the first logger call already lands after the compilation episode.

        Args:
            samples: ``(cumulative_step, perf_counter)`` pairs in call order.

        Returns:
            Per-chunk step times in seconds, empty if there is too little data.
        """
        start = 1 if samples and samples[0][0] == 0 else 0
        step_times = []
        for (step0, time0), (step1, time1) in zip(
            samples[start:], samples[start + 1 :]
        ):
            if step1 > step0:
                step_times.append((time1 - time0) / (step1 - step0))
        return step_times

    def _measure_model_throughput(self, atoms: Any) -> list[float]:
        """Time single-structure model forward passes (energy + forces).

        For mlip ``ForceField`` models this times the pure network forward on a
        pre-built graph (excluding neighbour-list construction), mirroring mlip-jax's
        ``scripts/time_inference.py``. For external ASE calculators it times a forced
        recomputation on the pre-built atoms (which includes the calculator's own
        neighbour-list build, as there is no jittable forward to isolate). Warm-up
        passes absorb compilation, the result is read to force device synchronisation,
        and the slowest `FORWARD_TRIM_FRACTION` of passes are dropped before the caller
        averages.

        Args:
            atoms: The structure to run inference on.

        Returns:
            The kept per-pass durations in seconds, or an empty list if it failed.
        """
        try:
            forward = self._build_forward_fn(atoms)

            for _ in range(NUM_FORWARD_WARMUP):
                forward()

            times = []
            for _ in range(self._num_forward_timed):
                start = time.perf_counter()
                forward()
                times.append(time.perf_counter() - start)

            # Drop the slowest passes (garbage-collection / scheduling spikes).
            times.sort()
            keep = max(1, round((1.0 - FORWARD_TRIM_FRACTION) * len(times)))
            return times[:keep]

        except Exception as e:
            logger.info(
                "Error measuring model throughput on system %s: %s", str(atoms), str(e)
            )
            return []

    def _build_forward_fn(self, atoms: Any) -> Any:
        """Build a zero-argument closure that runs (and blocks on) one forward pass.

        Args:
            atoms: The structure to run inference on.

        Returns:
            A callable taking no arguments that performs one forward pass.
        """
        from mlip.models import ForceField  # noqa: PLC0415

        if isinstance(self.force_field, ForceField):
            # mlip model: time the pure network forward on a pre-built graph.
            import jax  # noqa: PLC0415
            from mlip.data.chemical_system import ChemicalSystem  # noqa: PLC0415
            from mlip.graph import Graph  # noqa: PLC0415

            force_field = self.force_field
            chem_system = ChemicalSystem.from_ase_atoms(
                atoms, get_property_fields=False
            )
            graph = Graph.from_chemical_system(
                chem_system,
                force_field.cutoff_distance,
                long_range_cutoff_angstrom=force_field.long_range_cutoff_distance,
            )
            jitted = jax.jit(force_field)

            def mlip_forward() -> None:
                prediction = jitted(graph)
                jax.block_until_ready(prediction.forces)

            return mlip_forward

        # External ASE calculator: force a full recompute each call (ASE caches results
        # for an unchanged system, so we invalidate via system_changes=all_changes).
        from ase.calculators.calculator import all_changes  # noqa: PLC0415

        calculator = self.force_field

        def ase_forward() -> None:
            calculator.calculate(atoms, ["energy", "forces"], all_changes)
            _ = calculator.results["forces"]  # read to force device synchronisation

        return ase_forward

    def analyze(self) -> InferenceSpeedResult:
        """Aggregate the timings and compute the throughput score.

        Returns:
            An `InferenceSpeedResult` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        timestep_fs = float(self._md_kwargs["timestep_fs"])

        structure_results = []
        for i, structure_name in enumerate(self._structure_names):
            forward_times = (
                self.model_output.forward_times[i]
                if i < len(self.model_output.forward_times)
                else []
            )
            backend_step_times = (
                self.model_output.md_step_times[i]
                if i < len(self.model_output.md_step_times)
                else {}
            )

            average_forward_time = (
                sum(forward_times) / len(forward_times) if forward_times else None
            )

            md = {}
            for backend, step_times in backend_step_times.items():
                average_step_time = (
                    sum(step_times) / len(step_times) if step_times else None
                )
                md[backend] = MDBackendResult(
                    average_step_time=average_step_time,
                    step_time_samples=step_times,
                )

            any_md = any(b.average_step_time is not None for b in md.values())
            structure_results.append(
                InferenceSpeedStructureResult(
                    structure_name=structure_name,
                    num_atoms=get_molecule_size_from_name(structure_name),
                    num_steps=self._md_kwargs["num_steps"],
                    num_episodes=self._md_kwargs["num_episodes"],
                    timestep_fs=timestep_fs,
                    average_forward_time=average_forward_time,
                    forward_times=forward_times,
                    md=md,
                    failed=average_forward_time is None and not any_md,
                )
            )

        # The whole benchmark is failed if there are no structures at all (empty data
        # dir) or every structure failed; otherwise the model gets scored on the
        # structures that succeeded.
        if not structure_results or all(r.failed for r in structure_results):
            return InferenceSpeedResult(
                structure_names=self._structure_names,
                structures=structure_results,
                score=0.0,
                failed=True,
            )

        return InferenceSpeedResult(
            structure_names=self._structure_names,
            structures=structure_results,
            score=self._compute_score(structure_results),
            graph_cutoff_angstrom=self._graph_cutoff_angstrom(),
        )

    def _graph_cutoff_angstrom(self) -> float | None:
        """Best-effort retrieval of the model's graph cutoff in Angstrom.

        mlip ``ForceField`` models expose ``cutoff_distance``; external ASE
        calculators vary, so we probe a few common attribute names and fall back to
        None when none are present.

        Returns:
            The graph cutoff in Angstrom, or None if it cannot be determined.
        """
        for attr in ("cutoff_distance", "cutoff", "r_max"):
            value = getattr(self.force_field, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _compute_score(
        structure_results: list[InferenceSpeedStructureResult],
    ) -> float:
        """Score speed via a Hill function on the per-atom model forward time.

        Each structure contributes ``1 / (1 + (t / t0) ** k)`` where ``t`` is its
        per-atom forward-pass time (size-normalised so a single midpoint is meaningful
        across system sizes); structures with no successful forward pass score 0. The
        benchmark score is the mean. The forward pass is used (rather than the MD step)
        because it is engine-independent.

        Args:
            structure_results: The per-structure results.

        Returns:
            The mean speed score in [0, 1].
        """
        per_atom_forward_times = [
            r.average_forward_time / r.num_atoms
            if r.average_forward_time is not None
            else None
            for r in structure_results
        ]
        scores = compute_speed_score(
            per_atom_forward_times,
            midpoint=SCORE_PER_ATOM_FORWARD_TIME_MIDPOINT,
            sharpness=SCORE_SHARPNESS,
        )
        return float(scores.mean())

    @functools.cached_property
    def _structure_filenames(self) -> list[str]:
        structure_names = sorted(
            os.listdir(self.data_dir),
            key=get_molecule_size_from_name,
        )
        if self.run_mode == RunMode.DEV:
            return structure_names[:NUM_DEV_SYSTEMS]
        return structure_names

    @functools.cached_property
    def _structure_names(self) -> list[str]:
        return [Path(filename).stem for filename in self._structure_filenames]

    @functools.cached_property
    def _md_kwargs(self) -> dict[str, Any]:
        return (
            SIMULATION_CONFIG_DEV if self.run_mode == RunMode.DEV else SIMULATION_CONFIG
        )

    @functools.cached_property
    def _num_forward_timed(self) -> int:
        return (
            NUM_FORWARD_TIMED_DEV if self.run_mode == RunMode.DEV else NUM_FORWARD_TIMED
        )

    @functools.cached_property
    def _md_backends(self) -> list[str]:
        """MD backends to run: JAX-MD and ASE for mlip models, ASE only otherwise."""
        from mlip.models import ForceField  # noqa: PLC0415

        if isinstance(self.force_field, ForceField):
            return [JAX_MD_BACKEND, ASE_BACKEND]
        return [ASE_BACKEND]
