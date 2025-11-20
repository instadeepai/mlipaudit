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
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.utils.simulation import get_simulation_engine

SIMULATION_CONFIG = {
    "num_steps": 1000,
    "snapshot_interval": 100,
    "num_episodes": 5,
    "timestep_fs": 1,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "num_episodes": 10,
    "timestep_fs": 1,
}
NUM_DEV_SYSTEMS = 2

logger = logging.getLogger("mlipaudit")


class ScalingModelOutput(ModelOutput):
    """Model output for the scaling benchmark.

    Attributes:
        structure_names: The names of the structures used.
        simulation_states: A list of final simulation states for
            each corresponding structure. None if the simulation
            failed.
        average_episode_times: A list of average episode times
            for each corresponding structure, excluding the first
            episode to ignore the compilation time. None if the
            simulation failed.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]
    average_episode_times: list[float | None]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ScalingStructureResult(BaseModel):
    """Result object for a single structure.

    Attributes:
        structure_name: The structure name.
        num_atoms: The number of atoms in the structure.
        num_steps: The number of steps in the simulation.
        num_episodes: The number of episodes in the simulation.
        average_episode_time: The average episode time of the simulation,
            excluding the first episode to ignore the compilation time.
        average_step_time: The average step time of the simulation,
            excluding the first episode to ignore the compilation time.
        failed: Whether the simulation failed.
    """

    structure_name: str
    num_atoms: PositiveInt
    num_steps: PositiveInt
    num_episodes: PositiveInt
    average_episode_time: NonNegativeFloat | None = None
    average_step_time: NonNegativeFloat | None = None

    failed: bool = False


class ScalingResult(BenchmarkResult):
    """Result object for the scaling benchmark.

    Attributes:
        structure_names: The names of the structures.
        structures: List of per structure results.
    """

    structure_names: list[str]
    structures: list[ScalingStructureResult]


def get_molecule_size_from_name(name: str) -> int:
    """Get the molecule size from the name.

    Args:
        name: The name of the structure.

    Returns:
        The number of atoms in the structure.
    """
    return int(name.split("_", maxsplit=1)[0])


class Timer:
    """Track simulation episode times."""

    def __init__(self):
        """Constructor."""
        self.current_episode = 0
        self.episode_times = []
        self.call_time = 0.0

    def log(self, state: SimulationState) -> None:
        """Update the list of episode times.

        Args:
            state: The current simulation state.
        """
        if self.current_episode == 0:
            self.call_time = time.perf_counter()
        else:
            episode_duration = time.perf_counter() - self.call_time
            self.call_time = time.perf_counter()
            self.episode_times.append(episode_duration)
        self.current_episode += 1

    @property
    def average_episode_time(self) -> float:
        """The average episode time excluding the first episode.

        Raises:
            ValueError: If called but not episode times have been logged.
                If for instance accessed before running the engine.
        """
        if not self.episode_times:
            raise ValueError("No episode times available.")
        return sum(self.episode_times) / len(self.episode_times)


class ScalingBenchmark(Benchmark):
    """Benchmark for testing how inference speed scales.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `scaling`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category matches the default ("General").
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `ScalingResult`.
        model_output_class: A reference to the `ScalingModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "scaling"
    category = "General"
    result_class = ScalingResult
    model_output_class = ScalingModelOutput

    required_elements = {"N", "H", "O", "S", "P", "C"}

    def run_model(self) -> None:
        """Runs a short MD simulation for each structure, timing each
        episode and calculating the average episode time, ignoring the
        first to ignore the compilation time.
        """
        simulation_states: list[SimulationState | None] = []
        average_episode_times: list[float | None] = []
        for structure_name in self._structure_names:
            try:
                timer = Timer()
                atoms = ase_read(
                    self.data_input_dir / self.name / f"{structure_name}.xyz"
                )
                md_engine = get_simulation_engine(
                    atoms=atoms,
                    force_field=self.force_field,
                    **self._md_kwargs,
                )

                md_engine.attach_logger(timer.log)
                md_engine.run()

                simulation_states.append(md_engine.state)
                average_episode_times.append(timer.average_episode_time)

            except Exception as e:
                logger.info(
                    "Error running simulation on system %s: %s", str(atoms), str(e)
                )
                simulation_states.append(None)
                average_episode_times.append(None)

        self.model_output = ScalingModelOutput(
            structure_names=self._structure_names,
            simulation_states=simulation_states,
            average_episode_times=average_episode_times,
        )

    def analyze(self) -> ScalingResult:
        """Aggregate the average episode times and metadata.

        Returns:
            A `ScalingResult` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        structure_results = []
        for i, structure_name in enumerate(self._structure_names):
            if self.model_output.average_episode_times[i] is None:
                structure_results.append(
                    ScalingStructureResult(
                        structure_name=structure_name,
                        num_atoms=get_molecule_size_from_name(structure_name),
                        num_steps=self._md_kwargs["num_steps"],
                        num_episodes=self._md_kwargs["num_episodes"],
                        failed=True,
                    )
                )

            num_steps_per_episode = (
                self._md_kwargs["num_steps"] // self._md_kwargs["num_episodes"]
            )
            average_episode_time = self.model_output.average_episode_times[i]
            average_step_time = average_episode_time / num_steps_per_episode
            structure_results.append(
                ScalingStructureResult(
                    structure_name=structure_name,
                    num_atoms=get_molecule_size_from_name(structure_name),
                    num_steps=self._md_kwargs["num_steps"],
                    num_episodes=self._md_kwargs["num_episodes"],
                    average_episode_time=average_episode_time,
                    average_step_time=average_step_time,
                )
            )

        if len(self.model_output.simulation_states) == 0:
            return ScalingResult(structure_names=self._structure_names, failed=True)

        return ScalingResult(
            structure_names=self._structure_names, structures=structure_results
        )

    @functools.cached_property
    def _structure_filenames(self) -> list[str]:
        structure_names = sorted(
            os.listdir(self.data_input_dir / self.name), key=get_molecule_size_from_name
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
