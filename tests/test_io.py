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

import jax.numpy as jnp
import numpy as np
from mlip.simulation import SimulationState

from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    BondLengthDistributionBenchmark,
    BondLengthDistributionModelOutput,
    MoleculeModelOutput,
)
from mlipaudit.io import (
    load_benchmark_result_from_disk,
    load_benchmark_results_from_disk,
    load_model_output_from_disk,
    write_benchmark_result_to_disk,
    write_model_output_to_disk,
)


def test_benchmark_results_io_works(
    tmp_path,
    dummy_benchmark_results_model_1,
    dummy_benchmark_results_model_2,
    dummy_benchmark_1_class,
    all_dummy_benchmark_classes,
):
    """Tests whether results can be saved and loaded again to and from disk."""
    model_1_path = tmp_path / "model_1"
    for benchmark_name, result in dummy_benchmark_results_model_1.items():
        write_benchmark_result_to_disk(benchmark_name, result, model_1_path)

    assert set(os.listdir(model_1_path)) == {"benchmark_1", "benchmark_2"}
    assert os.listdir(model_1_path / "benchmark_1") == ["result.json"]

    model_2_path = tmp_path / "model_2"
    for benchmark_name, result in dummy_benchmark_results_model_2.items():
        write_benchmark_result_to_disk(benchmark_name, result, model_2_path)

    assert set(os.listdir(model_2_path)) == {"benchmark_1", "benchmark_2"}
    assert os.listdir(model_2_path / "benchmark_1") == ["result.json"]
    assert set(os.listdir(tmp_path)) == {"model_1", "model_2"}

    loaded_results = load_benchmark_results_from_disk(
        tmp_path, all_dummy_benchmark_classes
    )

    assert set(loaded_results.keys()) == {"model_1", "model_2"}
    assert loaded_results["model_1"] == dummy_benchmark_results_model_1
    assert loaded_results["model_2"] == dummy_benchmark_results_model_2

    # Test loading a single benchmark result
    loaded_single_result = load_benchmark_result_from_disk(
        model_1_path, dummy_benchmark_1_class
    )
    assert loaded_single_result == dummy_benchmark_results_model_1["benchmark_1"]


def test_model_outputs_io_works(
    tmp_path,
    dummy_model_output_class,
    dummy_subclass_model_output_class,
    dummy_benchmark_1_class,
    dummy_benchmark_2_class,
):
    """Tests whether model outputs can be saved and loaded again to and from disk."""
    # First, set up two different simulation states
    dummy_sim_state_1 = SimulationState(
        atomic_numbers=np.array([1, 8, 6, 1]),
        positions=jnp.ones((7, 4, 3)),
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
        "benchmark_1": dummy_model_output_class(
            structure_names=["s1", "s2", "s3"],
            simulation_states=[
                deepcopy(dummy_sim_state_1),
                deepcopy(dummy_sim_state_2),
                deepcopy(dummy_sim_state_1),
            ],
            subclasses=[
                dummy_subclass_model_output_class(
                    name="a", state=deepcopy(dummy_sim_state_1)
                )
            ],
        ),
        "benchmark_2": dummy_model_output_class(
            structure_names=["s4"],
            simulation_states=[deepcopy(dummy_sim_state_1)],
            subclasses=[
                dummy_subclass_model_output_class(
                    name="b", state=deepcopy(dummy_sim_state_2)
                ),
                dummy_subclass_model_output_class(
                    name="c", state=deepcopy(dummy_sim_state_1)
                ),
                dummy_subclass_model_output_class(
                    name="d", state=deepcopy(dummy_sim_state_2)
                ),
            ],
        ),
    }

    for benchmark_name, model_output in model_outputs.items():
        write_model_output_to_disk(benchmark_name, model_output, tmp_path)

    assert set(os.listdir(tmp_path)) == {"benchmark_1", "benchmark_2"}

    loaded_output_1 = load_model_output_from_disk(tmp_path, dummy_benchmark_1_class)
    loaded_output_2 = load_model_output_from_disk(tmp_path, dummy_benchmark_2_class)
    loaded_outputs = [loaded_output_1, loaded_output_2]

    for idx, model_output in enumerate(loaded_outputs):
        benchmark_name = "benchmark_1" if idx == 0 else "benchmark_2"
        assert isinstance(model_output, dummy_model_output_class)
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
            assert isinstance(subclass_1, dummy_subclass_model_output_class)
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


def test_loading_empty_simulation_states(tmp_path):
    """Test that if we save a model output that has a list
    of simulation states that are None, that it is correctly
    reloaded.
    """
    dummy_sim_state_1 = SimulationState(
        atomic_numbers=jnp.array([1, 8, 6, 1]),
        positions=jnp.ones((7, 4, 3)),
        forces=np.random.rand(7, 4, 3),
        velocities=jnp.zeros((7, 4, 3)),
        temperature=jnp.full((7,), 1.23),
        kinetic_energy=None,
        step=7,
        compute_time_seconds=42.7,
    )
    model_output = BondLengthDistributionModelOutput(
        molecules=[
            MoleculeModelOutput(molecule_name="1", simulation_state=dummy_sim_state_1),
            MoleculeModelOutput(molecule_name="2", simulation_state=None),
        ],
        num_failed=1,
    )
    model_1_path = tmp_path / "model_1"
    write_model_output_to_disk("bond_length_distribution", model_output, model_1_path)

    loaded_model_output = load_model_output_from_disk(
        model_1_path, BondLengthDistributionBenchmark
    )

    assert loaded_model_output.molecules[0].simulation_state is not None
    assert loaded_model_output.molecules[1].simulation_state is None
    assert loaded_model_output.num_failed == 1
