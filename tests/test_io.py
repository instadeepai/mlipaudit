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

import os
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pydantic
from mlip.simulation import SimulationState
from pydantic import ConfigDict

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.io import (
    load_benchmark_result_from_disk,
    load_benchmark_results_from_disk,
    load_model_output_from_disk,
    write_benchmark_result_to_disk,
    write_model_output_to_disk,
)


class DummyBenchmarkResultLarge(BenchmarkResult):
    """A dummy benchmark result with 5 entries."""

    a: int
    b: str
    c: list[float]
    d: list[tuple[float, float]]


class DummyBenchmarkResultSmallSubclass(BenchmarkResult):
    """A dummy benchmark result subclass."""

    value: float


class DummyBenchmarkResultSmall(BenchmarkResult):
    """A dummy benchmark result with one entry."""

    values: list[DummyBenchmarkResultSmallSubclass]


class DummySubclassModelOutput(pydantic.BaseModel):
    """A dummy model output subclass used in the other model output."""

    name: str
    state: SimulationState

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DummyModelOutput(ModelOutput):
    """A dummy model output class."""

    structure_names: list[str]
    simulation_states: list[SimulationState]
    subclasses: list[DummySubclassModelOutput]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DummyBenchmark1(Benchmark):
    """Dummy benchmark 1."""

    name = "benchmark_1"
    result_class = DummyBenchmarkResultLarge
    model_output_class = DummyModelOutput

    required_elements = {"H", "O"}

    def run_model(self) -> None:
        """No need to implement this for this test."""
        pass

    def analyze(self) -> DummyBenchmarkResultLarge:  # type:ignore
        """No need to implement this for this test."""
        pass


class DummyBenchmark2(Benchmark):
    """Dummy benchmark 2."""

    name = "benchmark_2"
    result_class = DummyBenchmarkResultSmall
    model_output_class = DummyModelOutput

    required_elements = {"H", "O"}

    def run_model(self) -> None:
        """No need to implement this for this test."""
        pass

    def analyze(self) -> list[DummyBenchmarkResultSmall]:  # type:ignore
        """No need to implement this for this test."""
        pass


def test_benchmark_results_io_works(tmpdir):
    """Tests whether results can be saved and loaded again to and from disk."""
    results_model_1 = {
        "benchmark_1": DummyBenchmarkResultLarge(
            a=7, b="test", c=[3.4, 5.6, 7.8], d=[(1.0, 1.1), (1.2, 1.3)]
        ),
        "benchmark_2": DummyBenchmarkResultSmall(
            values=[
                DummyBenchmarkResultSmallSubclass(value=0.1),
                DummyBenchmarkResultSmallSubclass(value=0.2),
            ]
        ),
    }

    results_model_2 = {
        "benchmark_1": DummyBenchmarkResultLarge(
            a=17, b="test", c=[13.4, 15.6, 17.8], d=[(11.0, 11.1), (11.2, 11.3)]
        ),
        "benchmark_2": DummyBenchmarkResultSmall(
            values=[
                DummyBenchmarkResultSmallSubclass(value=10.1),
                DummyBenchmarkResultSmallSubclass(value=10.2),
            ]
        ),
    }

    for name, result in results_model_1.items():
        write_benchmark_result_to_disk(name, result, Path(tmpdir) / "model_1")

    assert set(os.listdir(Path(tmpdir) / "model_1")) == {"benchmark_1", "benchmark_2"}
    assert os.listdir(Path(tmpdir) / "model_1" / "benchmark_1") == ["result.json"]

    for name, result in results_model_2.items():
        write_benchmark_result_to_disk(name, result, Path(tmpdir) / "model_2")

    assert set(os.listdir(Path(tmpdir) / "model_2")) == {"benchmark_1", "benchmark_2"}
    assert os.listdir(Path(tmpdir) / "model_2" / "benchmark_1") == ["result.json"]
    assert set(os.listdir(Path(tmpdir))) == {"model_1", "model_2"}

    loaded_results = load_benchmark_results_from_disk(
        tmpdir, [DummyBenchmark1, DummyBenchmark2]
    )

    assert set(loaded_results.keys()) == {"model_1", "model_2"}
    assert loaded_results["model_1"] == results_model_1
    assert loaded_results["model_2"] == results_model_2

    # Test that also the function that only loads one benchmark result works
    loaded_result = load_benchmark_result_from_disk(tmpdir / "model_1", DummyBenchmark1)
    assert loaded_result == results_model_1["benchmark_1"]


def test_model_outputs_io_works(tmpdir):
    """Tests whether model outputs can be saved and loaded again to and from disk."""
    # First, set up two different simulation states
    dummy_sim_state_1 = SimulationState(
        atomic_numbers=np.array([1, 8, 6, 1]),
        positions=np.ones((7, 4, 3)),
        forces=np.random.rand(7, 4, 3),
        velocities=np.zeros((7, 4, 3)),
        temperature=np.full((7,), 1.23),
        kinetic_energy=None,
        step=7,
        compute_time_seconds=42.7,
    )
    dummy_sim_state_2 = deepcopy(dummy_sim_state_1)
    dummy_sim_state_2.temperature = np.full((7,), 11.23)

    # Second, set up the model outputs dictionary with two benchmark outputs
    model_outputs = {
        "benchmark_1": DummyModelOutput(
            structure_names=["s1", "s2", "s3"],
            simulation_states=[
                deepcopy(dummy_sim_state_1),
                deepcopy(dummy_sim_state_2),
                deepcopy(dummy_sim_state_1),
            ],
            subclasses=[
                DummySubclassModelOutput(name="a", state=deepcopy(dummy_sim_state_1))
            ],
        ),
        "benchmark_2": DummyModelOutput(
            structure_names=["s4"],
            simulation_states=[deepcopy(dummy_sim_state_1)],
            subclasses=[
                DummySubclassModelOutput(name="b", state=deepcopy(dummy_sim_state_2)),
                DummySubclassModelOutput(name="c", state=deepcopy(dummy_sim_state_1)),
                DummySubclassModelOutput(name="d", state=deepcopy(dummy_sim_state_2)),
            ],
        ),
    }

    for name, model_output in model_outputs.items():
        write_model_output_to_disk(name, model_output, Path(tmpdir))

    assert set(os.listdir(Path(tmpdir))) == {"benchmark_1", "benchmark_2"}

    loaded_output_1 = load_model_output_from_disk(tmpdir, DummyBenchmark1)
    loaded_output_2 = load_model_output_from_disk(tmpdir, DummyBenchmark2)
    loaded_outputs = [loaded_output_1, loaded_output_2]

    for idx, model_output in enumerate(loaded_outputs):
        benchmark_name = "benchmark_1" if idx == 0 else "benchmark_2"
        assert isinstance(model_output, DummyModelOutput)
        assert (
            model_output.structure_names
            == model_outputs[benchmark_name].structure_names
        )
        assert len(model_output.simulation_states) == len(
            model_outputs[benchmark_name].simulation_states
        )
        assert len(model_output.subclasses) == len(
            model_outputs[benchmark_name].subclasses
        )

        for sim_state_1, sim_state_2 in zip(
            model_output.simulation_states,
            model_outputs[benchmark_name].simulation_states,
        ):
            for field in fields(SimulationState):
                if field.name in ("kinetic_energy", "step", "compute_time_seconds"):
                    assert getattr(sim_state_1, field.name) == getattr(
                        sim_state_2, field.name
                    )
                else:
                    assert np.array_equal(
                        getattr(sim_state_1, field.name),
                        getattr(sim_state_2, field.name),
                    )

        for subclass_1, subclass_2 in zip(
            model_output.subclasses, model_outputs[benchmark_name].subclasses
        ):
            assert isinstance(subclass_1, DummySubclassModelOutput)
            assert subclass_1.name == subclass_2.name
            for field in fields(SimulationState):
                if field.name in ("kinetic_energy", "step", "compute_time_seconds"):
                    assert getattr(subclass_1.state, field.name) == getattr(
                        subclass_2.state, field.name
                    )
                else:
                    assert np.array_equal(
                        getattr(subclass_1.state, field.name),
                        getattr(subclass_2.state, field.name),
                    )
