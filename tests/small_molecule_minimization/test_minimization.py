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
from ase.symbols import symbols2numbers
from mlip.simulation import SimulationState

# Import the base class as well to help with mocking
from mlipaudit.benchmarks.small_molecule_minimization.small_molecule_minimization import (  # noqa: E501
    MoleculeSimulationOutput,
    SmallMoleculeMinimizationBenchmark,
    SmallMoleculeMinimizationModelOutput,
    SmallMoleculeMinimizationResult,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def small_mol_minimization_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> SmallMoleculeMinimizationBenchmark:
    """Assembles a fully configured and isolated SmallMoleculeMinimizationBenchmark
    instance. This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized SmallMoleculeMinimizationBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return SmallMoleculeMinimizationBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


def _generate_fake_model_output() -> SmallMoleculeMinimizationModelOutput:
    def _gen_sim_output(mol_name: str, num_atoms: int) -> MoleculeSimulationOutput:
        return MoleculeSimulationOutput(
            molecule_name=mol_name,
            simulation_state=SimulationState(
                positions=np.ones((10, num_atoms, 3)), temperature=np.ones(10)
            ),
        )

    return SmallMoleculeMinimizationModelOutput(
        qm9_neutral=[_gen_sim_output("mol_0", 17), _gen_sim_output("mol_1", 4)],
        qm9_charged=[_gen_sim_output("mol_0", 16), _gen_sim_output("mol_1", 16)],
        openff_neutral=[_gen_sim_output("mol_0", 45), _gen_sim_output("mol_1", 6)],
        openff_charged=[_gen_sim_output("mol_0", 43), _gen_sim_output("mol_1", 9)],
    )


@pytest.mark.parametrize(
    "small_mol_minimization_benchmark", [True, False], indirect=True
)
def test_full_run_with_mocked_engine(
    small_mol_minimization_benchmark, mock_jaxmd_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = small_mol_minimization_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        # Assert that the engine was initialized and run for each molecule
        num_molecules = (
            len(benchmark._qm9_neutral_dataset)
            + len(benchmark._qm9_charged_dataset)
            + len(benchmark._openff_neutral_dataset)
            + len(benchmark._openff_charged_dataset)
        )
        assert mock_engine_class.call_count == num_molecules
        assert mock_engine.run.call_count == num_molecules

        assert isinstance(benchmark.model_output, SmallMoleculeMinimizationModelOutput)
        assert (
            len(benchmark.model_output.qm9_neutral)
            + len(benchmark.model_output.qm9_charged)
            + len(benchmark.model_output.openff_neutral)
            + len(benchmark.model_output.openff_charged)
            == num_molecules
        )

        benchmark.model_output = _generate_fake_model_output()

        result = benchmark.analyze()
        assert type(result) is SmallMoleculeMinimizationResult
        assert len(result.qm9_neutral.rmsd_values) == 2


def test_analyze_raises_error_if_run_first(small_mol_minimization_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        small_mol_minimization_benchmark.analyze()


def test_good_agreement(small_mol_minimization_benchmark):
    """Check the analysis method."""
    benchmark = small_mol_minimization_benchmark
    benchmark.model_output = _generate_fake_model_output()

    mol0_qm_neutral_coordinates = np.array([  # Same as reference
        [0.138, -0.1154, 0.4398],
        [0.0632, 1.0279, 0.054],
        [1.2463, 1.9232, -0.0293],
        [1.4427, 2.9217, -1.1774],
        [0.416, 3.1387, -2.2752],
        [-0.8662, 3.4805, -1.7571],
        [-0.7633, 4.4593, -0.7615],
        [-0.1647, 3.9552, 0.4119],
        [1.113, 3.4298, 0.1893],
        [-0.8989, 1.4713, -0.2625],
        [2.1522, 1.4626, 0.3505],
        [2.4705, 3.0263, -1.51],
        [0.7742, 3.9502, -2.9288],
        [0.2728, 2.2466, -2.8907],
        [-0.1902, 5.3249, -1.1385],
        [-1.7759, 4.758, -0.4878],
        [1.8774, 3.908, 0.7961],  # H atom
    ])

    # Modifying H should affect output
    mol0_qm_neutral_coordinates[-1] += 1.0
    mol0_qm_neutral_coordinates = np.expand_dims(mol0_qm_neutral_coordinates, axis=0)

    mol1_qm_neutral_coordinates = np.array([  # Same as reference
        [0.0175, -1.1613, -0.0042],
        [0.0025, -0.0034, 0.0018],
        [-0.0161, 1.3722, 0.0093],
        [-0.0326, 2.5301, 0.0161],
    ])
    mol1_qm_neutral_coordinates = np.expand_dims(mol1_qm_neutral_coordinates, 0)
    benchmark.model_output.qm9_neutral = [
        MoleculeSimulationOutput(
            molecule_name="mol_0",
            simulation_state=SimulationState(
                atomic_numbers=np.array(
                    symbols2numbers([
                        "O",
                        "C",
                        "C",
                        "C",
                        "C",
                        "O",
                        "C",
                        "O",
                        "C",
                        "H",
                        "H",
                        "H",
                        "H",
                        "H",
                        "H",
                        "H",
                        "H",
                    ])
                ),
                positions=mol0_qm_neutral_coordinates,
                temperature=np.ones(10),
            ),
        ),
        MoleculeSimulationOutput(
            molecule_name="mol_1",
            simulation_state=SimulationState(
                atomic_numbers=np.array(symbols2numbers(["N", "C", "C", "N"])),
                positions=mol1_qm_neutral_coordinates,
                temperature=np.ones(10),
            ),
        ),
    ]

    result = benchmark.analyze()

    assert result.qm9_neutral.rmsd_values[0] is None
    assert result.qm9_neutral.avg_rmsd == result.qm9_neutral.rmsd_values[1]
    assert result.score

    assert (
        result.qm9_neutral.rmsd_values[1] < 1e-3
    )  # We actually get 6e-4 due to mdtraj implementation


def test_bad_agreement(small_mol_minimization_benchmark):
    """Check the analysis method."""
    benchmark = small_mol_minimization_benchmark
    benchmark.model_output = _generate_fake_model_output()

    mol1_qm_neutral_coordinates = np.array([  # Same as reference
        [0.0175, -1.1613, -0.0042],
        [0.0025, -0.0034, 0.0018],
        [-0.0161, 1.3722, 0.0093],
        [-0.0326, 2.5301, 0.0161],
    ])
    mol1_qm_neutral_coordinates[-1] += 0.2
    mol1_qm_neutral_coordinates = np.expand_dims(mol1_qm_neutral_coordinates, 0)
    benchmark.model_output.qm9_neutral[1] = MoleculeSimulationOutput(
        molecule_name="mol_1",
        simulation_state=SimulationState(
            atomic_numbers=np.array(symbols2numbers(["N", "C", "C", "N"])),
            positions=mol1_qm_neutral_coordinates,
            temperature=np.ones(10),
        ),
    )

    result = benchmark.analyze()
    assert result.qm9_neutral.rmsd_values[1] > 1e-3
