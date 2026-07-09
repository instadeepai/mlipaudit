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

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase.symbols import symbols2numbers
from mlip.models import ForceField
from mlip.simulation import SimulationState
from mlip.simulation.enums import MDIntegrator

from mlipaudit.benchmarks.nve_energy_conservation.nve_energy_conservation import (
    SYSTEMS_BY_NAME,
    NVEEnergyConservationBenchmark,
    NVEEnergyConservationResult,
)
from mlipaudit.run_mode import RunMode

DATA_DIR = Path(__file__).parent.parent / "data"

# `run_simulation` is imported into the benchmark module's namespace, so patch it there.
RUN_SIMULATION_PATH = (
    "mlipaudit.benchmarks.nve_energy_conservation."
    "nve_energy_conservation.run_simulation"
)


@pytest.fixture
def nve_benchmark(
    mocked_benchmark_init,  # prevents the HuggingFace data download
    mock_force_field,  # mock force field supporting H, C, N, O, ...
) -> NVEEnergyConservationBenchmark:
    """Assemble an isolated NVE benchmark in dev mode (vacuum system only).

    Returns:
        An initialized NVEEnergyConservationBenchmark instance.
    """
    return NVEEnergyConservationBenchmark(
        force_field=mock_force_field,
        data_input_dir=DATA_DIR,
        run_mode=RunMode.DEV,
    )


def test_analyze_raises_if_run_model_not_called(nve_benchmark):
    """analyze() must fail if called before run_model()."""
    with pytest.raises(RuntimeError):
        nve_benchmark.analyze()


def test_full_run_dev_with_mocked_simulation(nve_benchmark):
    """A dev run loads the vacuum system, runs NVE MD and analyzes it."""
    num_frames = 11
    kinetic = np.linspace(1.0, 1.2, num_frames)
    potential = np.linspace(-5.0, -5.2, num_frames)
    state = SimulationState(
        atomic_numbers=np.array([1, 6, 7, 8]),
        kinetic_energy=kinetic,
        potential_energy=potential,
    )

    with patch(RUN_SIMULATION_PATH, return_value=state) as mock_run_simulation:
        nve_benchmark.run_model()

    # A single (vacuum) system is run in dev mode, with the NVE integrator.
    assert mock_run_simulation.call_count == 1
    _, kwargs = mock_run_simulation.call_args
    assert kwargs["md_integrator"] == MDIntegrator.NVE_VELOCITY_VERLET
    assert "box" not in kwargs  # vacuum system has no periodic box

    output = nve_benchmark.model_output
    assert output.structure_names == ["Small_molecule_HCNO"]
    assert output.n_skipped_unallowed_elements == 0
    assert len(output.times_ps[0]) == num_frames
    assert len(output.potential_energies_ev[0]) == num_frames
    assert len(output.kinetic_energies_ev[0]) == num_frames

    result = nve_benchmark.analyze()
    assert isinstance(result, NVEEnergyConservationResult)
    assert not result.failed
    assert len(result.structure_results) == 1
    structure_result = result.structure_results[0]
    assert not structure_result.skipped
    assert not structure_result.failed
    assert structure_result.score is not None
    assert result.score is not None


def test_failed_simulation_is_recorded(nve_benchmark):
    """A failed simulation (None) yields a failed system that scores 0.0.

    The overall result is not flagged ``failed`` because a failed (non-skipped)
    system still contributes a valid score of 0.0; the ``failed`` flag is reserved
    for the case where no system produces a score at all.
    """
    with patch(RUN_SIMULATION_PATH, return_value=None):
        nve_benchmark.run_model()

    output = nve_benchmark.model_output
    assert output.times_ps[0] is None
    assert output.potential_energies_ev[0] is None

    result = nve_benchmark.analyze()
    assert not result.failed
    assert result.score == pytest.approx(0.0)
    assert result.structure_results[0].failed
    assert result.structure_results[0].score == pytest.approx(0.0)


def test_skips_systems_with_unallowed_elements(mocked_benchmark_init):
    """Systems whose elements the model cannot handle are skipped, not simulated."""
    force_field = MagicMock(spec=ForceField)
    # No carbon: the vacuum HCNO system cannot be simulated.
    force_field.allowed_atomic_numbers = symbols2numbers({"H", "N", "O"})

    benchmark = NVEEnergyConservationBenchmark(
        force_field=force_field,
        data_input_dir=DATA_DIR,
        run_mode=RunMode.DEV,
    )

    with patch(RUN_SIMULATION_PATH) as mock_run_simulation:
        benchmark.run_model()

    mock_run_simulation.assert_not_called()
    output = benchmark.model_output
    assert output.skipped_structures == ["Small_molecule_HCNO"]
    assert output.n_skipped_unallowed_elements == 1

    result = benchmark.analyze()
    assert result.failed  # every (selected) system was skipped
    assert result.n_skipped_unallowed_elements == 1
    assert result.structure_results[0].skipped


def test_analyze_structure_conserved_energy_scores_high(nve_benchmark):
    """A perfectly conserved trajectory (flat total energy) scores 1.0."""
    spec = SYSTEMS_BY_NAME["Small_molecule_HCNO"]
    num_frames = 11
    times = (np.arange(num_frames) * 0.01).tolist()
    kinetic = (1.0 + 0.1 * np.sin(np.arange(num_frames))).tolist()
    # Choose the potential so the total energy is exactly constant.
    potential = (-np.asarray(kinetic)).tolist()

    result = nve_benchmark._analyze_structure(spec, 4, times, potential, kinetic)

    assert not result.failed
    assert result.total_drift_ev == pytest.approx(0.0, abs=1e-9)
    assert result.energy_drift_ratio == pytest.approx(0.0, abs=1e-9)
    assert result.score == pytest.approx(1.0)


def test_analyze_structure_drifting_energy_scores_low(nve_benchmark):
    """A strongly drifting trajectory scores well below 1.0."""
    spec = SYSTEMS_BY_NAME["Small_molecule_HCNO"]
    num_frames = 11
    times = np.arange(num_frames) * 0.01
    kinetic = 1.0 + 0.05 * np.sin(np.arange(num_frames))  # small fluctuation scale
    total = 10.0 * times  # large linear drift
    potential = total - kinetic

    result = nve_benchmark._analyze_structure(
        spec, 4, times.tolist(), potential.tolist(), kinetic.tolist()
    )

    assert not result.failed
    assert result.drift_slope_ev_per_ps == pytest.approx(10.0, rel=1e-3)
    assert result.energy_drift_ratio > 1.0
    assert result.score < 0.5


@pytest.mark.parametrize(
    "times, potential, kinetic",
    [
        (None, None, None),  # skipped / failed simulation
        ([0.0], [1.0], [1.0]),  # fewer than two points
        ([0.0, 0.1, 0.2], [1.0, float("nan"), 1.0], [1.0, 1.0, 1.0]),  # non-finite
        ([0.0, 0.1, 0.2], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),  # zero KE fluctuation
    ],
)
def test_analyze_structure_failure_cases(nve_benchmark, times, potential, kinetic):
    """Degenerate trajectories are flagged as failed with a zero score."""
    spec = SYSTEMS_BY_NAME["Small_molecule_HCNO"]
    result = nve_benchmark._analyze_structure(spec, 4, times, potential, kinetic)
    assert result.failed
    assert result.score == 0.0
