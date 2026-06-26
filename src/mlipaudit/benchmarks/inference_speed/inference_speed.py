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

#: Score parameters for the Hill function ``score = 1 / (1 + (t / t0) ** k)``, where
#: ``t`` is the per-atom step time in seconds. ``SCORE_PER_ATOM_STEP_TIME_MIDPOINT``
#: (``t0``) is the per-atom step time that scores 0.5 and ``SCORE_SHARPNESS`` (``k``)
#: controls how sharply models separate. These are calibrated for our H100 reference
#: hardware; the score is only comparable across models run on the same hardware.
#: TODO: tune ``t0`` against real model runs once available.
SCORE_PER_ATOM_STEP_TIME_MIDPOINT = 1.0e-5
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
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]
    average_episode_times: list[float | None]
    episode_times: list[list[float]] = []

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
            first episode to ignore the compilation time.
        timestep_fs: The MD timestep in femtoseconds, used to convert step times into
            a throughput (ns/day).
        episode_times: The individual episode durations (excluding the first episode),
            used to quantify timing variance. Empty if unavailable.
        failed: Whether the simulation failed.
    """

    structure_name: str
    num_atoms: PositiveInt
    num_steps: PositiveInt
    num_episodes: PositiveInt
    average_episode_time: NonNegativeFloat | None = None
    average_step_time: NonNegativeFloat | None = None
    timestep_fs: float | None = None
    episode_times: list[float] = []

    failed: bool = False


class InferenceSpeedResult(BenchmarkResult):
    """Result object for the inference-speed benchmark.

    Attributes:
        structure_names: The names of the structures.
        structures: List of per structure results.
    """

    structure_names: list[str]
    structures: list[InferenceSpeedStructureResult]


class InferenceSpeedBenchmark(Benchmark):
    """Benchmark measuring MD throughput and how it scales with system size.

    Runs a short MD simulation for each structure (reusing the ``scaling`` dataset),
    measures the per-step wall-clock time, and scores the model with a Hill function on
    its per-atom step time so that faster models score higher. The score is wall-clock
    based and therefore only comparable across models run on the same hardware.

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
        """Runs a short MD simulation for each structure, timing each episode and
        recording the per-episode times, ignoring the first to discard compilation.
        """
        simulation_states: list[SimulationState | None] = []
        average_episode_times: list[float | None] = []
        episode_times: list[list[float]] = []
        for structure_name in self._structure_names:
            try:
                timer = Timer()
                atoms = ase_read(
                    self.data_input_dir / self._dataset_name / f"{structure_name}.xyz"
                )
                atoms.info["charge"] = DEFAULT_CHARGE
                atoms.info["spin"] = DEFAULT_SPIN
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
        )

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
            if self.model_output.average_episode_times[i] is None:
                structure_results.append(
                    InferenceSpeedStructureResult(
                        structure_name=structure_name,
                        num_atoms=get_molecule_size_from_name(structure_name),
                        num_steps=self._md_kwargs["num_steps"],
                        num_episodes=self._md_kwargs["num_episodes"],
                        timestep_fs=timestep_fs,
                        failed=True,
                    )
                )
                continue

            average_episode_time = self.model_output.average_episode_times[i]
            average_step_time = average_episode_time / num_steps_per_episode
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
        )

    @staticmethod
    def _compute_score(
        structure_results: list[InferenceSpeedStructureResult],
    ) -> float:
        """Score speed via a Hill function on the per-atom step time.

        Each structure contributes ``1 / (1 + (t / t0) ** k)`` where ``t`` is its
        per-atom step time (size-normalised so a single midpoint is meaningful across
        system sizes); failed structures score 0. The benchmark score is the mean.

        Args:
            structure_results: The per-structure results.

        Returns:
            The mean speed score in [0, 1].
        """
        per_atom_step_times = [
            r.average_step_time / r.num_atoms
            if not r.failed and r.average_step_time is not None
            else None
            for r in structure_results
        ]
        scores = compute_speed_score(
            per_atom_step_times,
            midpoint=SCORE_PER_ATOM_STEP_TIME_MIDPOINT,
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
