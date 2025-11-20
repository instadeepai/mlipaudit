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

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from mlip.simulation import SimulationState

# Import the base class as well to help with mocking
from mlipaudit.benchmarks.reference_geometry_stability.reference_geometry_stability import (  # noqa: E501
    MoleculeModelOutput,
    ReferenceGeometryStabilityBenchmark,
    ReferenceGeometryStabilityModelOutput,
    ReferenceGeometryStabilityResult,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def ref_geometry_stability_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> ReferenceGeometryStabilityBenchmark:
    """Assembles a fully configured and isolated ReferenceGeometryStabilityBenchmark
    instance. This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized ReferenceGeometryStabilityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return ReferenceGeometryStabilityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


def _generate_fake_model_output() -> ReferenceGeometryStabilityModelOutput:
    def _gen_sim_output(mol_name: str, num_atoms: int) -> MoleculeModelOutput:
        return MoleculeModelOutput(
            molecule_name=mol_name,
            simulation_state=SimulationState(
                positions=np.ones((10, num_atoms, 3)), temperature=np.ones(10)
            ),
        )

    return ReferenceGeometryStabilityModelOutput(
        openff_neutral=[_gen_sim_output("mol_0", 45), _gen_sim_output("mol_1", 6)],
        openff_charged=[_gen_sim_output("mol_0", 43), _gen_sim_output("mol_1", 9)],
    )


@pytest.mark.parametrize(
    "ref_geometry_stability_benchmark", [True, False], indirect=True
)
def test_full_run_with_mocked_engine(
    ref_geometry_stability_benchmark, mock_ase_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = ref_geometry_stability_benchmark
    mock_engine = mock_ase_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.ASESimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        # Assert that the engine was initialized and run for each molecule
        num_molecules = +len(benchmark._openff_neutral_dataset) + len(
            benchmark._openff_charged_dataset
        )
        assert mock_engine_class.call_count == num_molecules
        assert mock_engine.run.call_count == num_molecules

        assert isinstance(benchmark.model_output, ReferenceGeometryStabilityModelOutput)
        assert (
            +len(benchmark.model_output.openff_neutral)
            + len(benchmark.model_output.openff_charged)
            == num_molecules
        )

        benchmark.model_output = _generate_fake_model_output()

        result = benchmark.analyze()
        assert type(result) is ReferenceGeometryStabilityResult
        assert len(result.openff_neutral.rmsd_values) == 2


def test_analyze_raises_error_if_run_first(ref_geometry_stability_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        ref_geometry_stability_benchmark.analyze()


def test_good_agreement(ref_geometry_stability_benchmark):
    """Check the analysis method."""
    benchmark = ref_geometry_stability_benchmark
    benchmark.model_output = _generate_fake_model_output()

    mol0_openff_neutral_coordinates = np.array(  # same as reference
        [
            [-8.6127, -1.0243, -7.9257],
            [-7.3864, -0.4364, -7.4953],
            [-7.0532, -0.5013, -6.1746],
            [-5.8467, 0.0857, -5.82],
            [-5.4194, 0.0679, -4.475],
            [-4.2339, 0.6453, -4.1287],
            [-3.9017, 0.5897, -2.8551],
            [-2.718, 1.1637, -2.556],
            [-2.2699, 1.1854, -1.1794],
            [-0.7469, 1.3835, -1.2351],
            [-0.1988, 2.0445, 0.0495],
            [-1.0712, 1.6855, 1.2573],
            [-2.5002, 2.2517, 1.0883],
            [-2.9404, 2.3073, -0.3911],
            [-0.41, 2.1378, 2.5002],
            [-0.8289, 1.3897, 3.9574],
            [0.5715, 0.2893, 4.2254],
            [-2.0123, 0.5407, 3.7727],
            [-0.7929, 2.4414, 4.9801],
            [-4.6071, 0.0213, -1.8332],
            [-5.7537, -0.5401, -2.1663],
            [-6.246, -0.5625, -3.4904],
            [-7.4727, -1.154, -3.8775],
            [-7.8785, -1.1296, -5.1946],
            [-9.4757, -0.5483, -7.4454],
            [-8.6593, -0.8516, -9.0014],
            [-8.6297, -2.1032, -7.7316],
            [-5.2278, 0.5604, -6.574],
            [-2.5225, 0.2259, -0.7225],
            [-0.2647, 0.4149, -1.4025],
            [-0.5159, 2.0069, -2.1036],
            [-0.1843, 3.136, -0.0625],
            [0.8334, 1.7337, 0.2328],
            [-1.1256, 0.5949, 1.3281],
            [-3.1891, 1.6456, 1.6812],
            [-2.5295, 3.2633, 1.511],
            [-4.0262, 2.2152, -0.4648],
            [-2.6608, 3.2638, -0.8476],
            [-0.4556, 3.1462, 2.6466],
            [0.6399, -0.3965, 3.38],
            [0.3719, -0.2595, 5.1475],
            [1.4725, 0.8956, 4.3159],
            [-6.3278, -0.9991, -1.3597],
            [-8.0969, -1.631, -3.125],
            [-8.8198, -1.587, -5.475],
        ]
    )

    # Modifying H should affect output
    mol0_openff_neutral_coordinates[-1] += 1.0
    mol0_openff_neutral_coordinates = np.expand_dims(
        mol0_openff_neutral_coordinates, axis=0
    )

    mol1_openff_neutral_coordinates = np.array([  # Same as reference
        [0.9991, 1.0, -0.0],
        [2.3472, 1.0, -0.0],
        [0.0711, -0.4621, -0.0],
        [0.071, 2.4621, 0.0],
        [3.2752, -0.4621, 0.0],
        [3.2753, 2.4621, -0.0],
    ])
    mol1_openff_neutral_coordinates = np.expand_dims(mol1_openff_neutral_coordinates, 0)
    benchmark.model_output.openff_neutral = [
        MoleculeModelOutput(
            molecule_name="mol_0",
            simulation_state=SimulationState(
                atomic_numbers=[
                    "C",
                    "O",
                    "C",
                    "C",
                    "C",
                    "N",
                    "C",
                    "O",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "N",
                    "S",
                    "C",
                    "O",
                    "O",
                    "N",
                    "C",
                    "C",
                    "C",
                    "C",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                ],
                positions=mol0_openff_neutral_coordinates,
            ),
        ),
        MoleculeModelOutput(
            molecule_name="mol_1",
            simulation_state=SimulationState(
                atomic_numbers=["C", "C", "Cl", "Cl", "Cl", "Cl"],
                positions=mol1_openff_neutral_coordinates,
            ),
        ),
    ]

    result = benchmark.analyze()

    assert result.openff_neutral.rmsd_values[0] < 1e-2
    assert result.openff_neutral.rmsd_values[1] < 1e-2


def test_bad_agreement(ref_geometry_stability_benchmark):
    """Check the analysis method."""
    benchmark = ref_geometry_stability_benchmark
    benchmark.model_output = _generate_fake_model_output()

    mol1_openff_neutral_coordinates = np.array([  # Same as reference
        [0.9991, 1.0, -0.0],
        [2.3472, 1.0, -0.0],
        [0.0711, -0.4621, -0.0],
        [0.071, 2.4621, 0.0],
        [3.2752, -0.4621, 0.0],
        [3.2753, 2.4621, -0.0],
    ])
    mol1_openff_neutral_coordinates[-1] += 1.0
    mol1_openff_neutral_coordinates = np.expand_dims(mol1_openff_neutral_coordinates, 0)
    benchmark.model_output.openff_neutral[1] = MoleculeModelOutput(
        molecule_name="mol_1",
        simulation_state=SimulationState(
            atomic_numbers=["C", "C", "Cl", "Cl", "Cl", "Cl"],
            positions=mol1_openff_neutral_coordinates,
        ),
    )

    result = benchmark.analyze()
    assert result.openff_neutral.rmsd_values[0] > 1e-3
