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
system size, and turns it into a single throughput score. It reuses the ``scaling``
benchmark's size-stratified protein dataset and its timing helpers, but additionally
records the per-episode timing spread and the MD timestep, and produces a
hardware-relative speed score.
"""

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from mlipaudit.benchmark import (
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    Benchmark,
    BenchmarkResult,
    ModelOutput,
)
from mlipaudit.benchmarks.scaling.scaling import (
    NUM_DEV_SYSTEMS,
    SIMULATION_CONFIG,
    SIMULATION_CONFIG_DEV,
    Timer,
    get_molecule_size_from_name,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_speed_score
from mlipaudit.utils.simulation import get_simulation_engine

#: Number of forward passes to discard (JIT compilation / lazy init) and to time when
#: measuring model throughput. After timing, the slowest `FORWARD_TRIM_FRACTION` of
#: passes are dropped (garbage-collection / scheduling spikes) before averaging.
NUM_FORWARD_WARMUP = 2
NUM_FORWARD_TIMED = 25
NUM_FORWARD_TIMED_DEV = 3
FORWARD_TRIM_FRACTION = 0.2

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


class InferenceSpeedModelOutput(ModelOutput):
    """Model output for the inference-speed benchmark.

    Attributes:
        structure_names: The names of the structures used.
        simulation_states: A list of final simulation states for each corresponding
            structure. None if the simulation failed.
        average_episode_times: A list of average episode times for each corresponding
            structure, excluding the first episode to ignore the compilation time.
            None if the simulation failed.
        episode_times: A list, per structure, of the individual episode durations
            (excluding the first episode) used to quantify timing variance. Empty for
            structures whose simulation failed.
        forward_times: A list, per structure, of the individual timed model
            forward-pass durations (excluding warm-up). Empty for structures whose
            forward pass failed.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]
    average_episode_times: list[float | None]
    episode_times: list[list[float]] = []
    forward_times: list[list[float]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class InferenceSpeedStructureResult(BaseModel):
    """Result object for a single structure.

    Attributes:
        structure_name: The structure name.
        num_atoms: The number of atoms in the structure.
        num_steps: The number of steps in the simulation.
        num_episodes: The number of episodes in the simulation.
        average_episode_time: The average episode time of the simulation, excluding
            the first episode to ignore the compilation time.
        average_step_time: The average step time of the simulation, excluding the
            first episode to ignore the compilation time. None if the MD run failed.
        timestep_fs: The MD timestep in femtoseconds, used to convert step times into
            a throughput (ns/day).
        episode_times: The individual episode durations (excluding the first episode),
            used to quantify timing variance. Empty if unavailable.
        average_forward_time: The average wall-clock time of a single model forward
            pass (energy + forces), excluding warm-up. This is the engine-independent
            model-throughput metric. None if the forward pass failed.
        forward_times: The individual timed forward-pass durations, used to quantify
            variance. Empty if unavailable.
        failed: Whether both the MD run and the forward pass failed for this structure.
    """

    structure_name: str
    num_atoms: PositiveInt
    num_steps: PositiveInt
    num_episodes: PositiveInt
    average_episode_time: NonNegativeFloat | None = None
    average_step_time: NonNegativeFloat | None = None
    timestep_fs: float | None = None
    episode_times: list[float] = []
    average_forward_time: NonNegativeFloat | None = None
    forward_times: list[float] = []

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

    For each structure (reusing the ``scaling`` dataset) it measures two complementary
    speeds: the **model forward-pass** time (energy + forces, engine-independent) and
    the **MD step** time (end-to-end, including neighbour lists, the integrator and the
    simulation engine). The gap between them reflects simulation overhead. The model is
    scored with a Hill function on its per-atom forward-pass time so that faster models
    score higher; the score is wall-clock based and only comparable across models run on
    the same hardware.

    Attributes:
        name: The unique benchmark name (``inference_speed``).
        category: The benchmark category, used for grouping in the UI.
        dataset_name: Set to ``scaling`` so this benchmark reuses the ``scaling``
            dataset rather than shipping a duplicate.
        result_class: The `InferenceSpeedResult` type returned by `analyze`.
        model_output_class: The `InferenceSpeedModelOutput` type.
        required_elements: The element types present in the input files.
    """

    name = "inference_speed"
    category = "General"
    dataset_name = "scaling"
    result_class = InferenceSpeedResult
    model_output_class = InferenceSpeedModelOutput

    required_elements = {"N", "H", "O", "S", "C"}

    def run_model(self) -> None:
        """For each structure, time the model forward pass (model throughput) and run a
        short MD simulation (MD throughput). The two measurements fail independently.
        """
        simulation_states: list[SimulationState | None] = []
        average_episode_times: list[float | None] = []
        episode_times: list[list[float]] = []
        forward_times: list[list[float]] = []
        for structure_name in self._structure_names:
            try:
                atoms = ase_read(
                    self.data_input_dir / self._dataset_name / f"{structure_name}.xyz"
                )
                atoms.info["charge"] = DEFAULT_CHARGE
                atoms.info["spin"] = DEFAULT_SPIN
            except Exception as e:
                logger.info("Error reading system %s: %s", structure_name, str(e))
                simulation_states.append(None)
                average_episode_times.append(None)
                episode_times.append([])
                forward_times.append([])
                continue

            # Model throughput: timed forward passes on a copy to avoid side effects.
            forward_times.append(self._measure_model_throughput(atoms.copy()))

            # MD throughput: short MD simulation timed per episode.
            try:
                timer = Timer()
                md_engine = get_simulation_engine(
                    atoms=atoms,
                    force_field=self.force_field,
                    **self._md_kwargs,
                )
                md_engine.attach_logger(timer.log)
                md_engine.run()

                simulation_states.append(md_engine.state)
                average_episode_times.append(timer.average_episode_time)
                episode_times.append(timer.episode_times)

            except Exception as e:
                logger.info(
                    "Error running simulation on system %s: %s", str(atoms), str(e)
                )
                simulation_states.append(None)
                average_episode_times.append(None)
                episode_times.append([])

        self.model_output = InferenceSpeedModelOutput(
            structure_names=self._structure_names,
            simulation_states=simulation_states,
            average_episode_times=average_episode_times,
            episode_times=episode_times,
            forward_times=forward_times,
        )

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
        num_steps_per_episode = (
            self._md_kwargs["num_steps"] // self._md_kwargs["num_episodes"]
        )

        structure_results = []
        for i, structure_name in enumerate(self._structure_names):
            episode_times = (
                self.model_output.episode_times[i]
                if i < len(self.model_output.episode_times)
                else []
            )
            forward_times = (
                self.model_output.forward_times[i]
                if i < len(self.model_output.forward_times)
                else []
            )

            average_episode_time = self.model_output.average_episode_times[i]
            average_step_time = (
                average_episode_time / num_steps_per_episode
                if average_episode_time is not None
                else None
            )
            average_forward_time = (
                sum(forward_times) / len(forward_times) if forward_times else None
            )

            structure_results.append(
                InferenceSpeedStructureResult(
                    structure_name=structure_name,
                    num_atoms=get_molecule_size_from_name(structure_name),
                    num_steps=self._md_kwargs["num_steps"],
                    num_episodes=self._md_kwargs["num_episodes"],
                    average_episode_time=average_episode_time,
                    average_step_time=average_step_time,
                    timestep_fs=timestep_fs,
                    episode_times=episode_times,
                    average_forward_time=average_forward_time,
                    forward_times=forward_times,
                    failed=average_step_time is None and average_forward_time is None,
                )
            )

        if len(self.model_output.simulation_states) == 0:
            return InferenceSpeedResult(
                structure_names=self._structure_names, failed=True
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
            os.listdir(self.data_input_dir / self._dataset_name),
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
