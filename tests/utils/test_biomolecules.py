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

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase.io import read as ase_read
from mlip.simulation import SimulationState

from mlipaudit.run_mode import RunMode
from mlipaudit.utils.biomolecules import iter_biomolecule_simulations

DATA_DIR = Path(__file__).parent.parent / "data" / "folding_stability"
STRUCTURE = "chignolin_1uao_xray"


def _simulation_state(atoms, positions: np.ndarray) -> SimulationState:
    num_frames = positions.shape[0]
    return SimulationState(
        atomic_numbers=atoms.numbers,
        positions=positions,
        forces=np.zeros_like(positions),
        temperature=np.zeros(num_frames),
    )


@pytest.mark.parametrize("failure_mode", ["crashed", "unstable"])
def test_failed_minimization_warns_and_runs_md_from_input(failure_mode, caplog):
    """A crashed (None) or blown-up minimization must not seed the MD.

    Instead, the MD should start from the un-minimized input structure and a
    warning should be emitted so the fallback is not silent.
    """
    atoms = ase_read(DATA_DIR / f"{STRUCTURE}.xyz")
    input_positions = atoms.get_positions()
    num_atoms = len(atoms)

    if failure_mode == "crashed":
        minimization_state = None
    else:  # exploded minimization -> NaNs in the final frame
        nan_positions = np.full((1, num_atoms, 3), np.nan)
        minimization_state = _simulation_state(atoms, nan_positions)

    md_state = _simulation_state(atoms, np.random.rand(2, num_atoms, 3))

    seen_md_positions = []

    def fake_run_simulation(atoms_arg, force_field, **kwargs):
        if kwargs.get("simulation_type") == "minimization":
            return minimization_state
        # Record the coordinates the MD is actually seeded with.
        seen_md_positions.append(atoms_arg.get_positions().copy())
        return md_state

    with patch(
        "mlipaudit.utils.biomolecules.run_simulation",
        side_effect=fake_run_simulation,
    ):
        with caplog.at_level(logging.WARNING):
            results = list(
                iter_biomolecule_simulations(MagicMock(), DATA_DIR, RunMode.DEV)
            )

    # One system (DEV), MD ran and its state was yielded.
    assert results == [(STRUCTURE, md_state)]
    # The MD was seeded from the input structure, not the failed minimization.
    np.testing.assert_allclose(seen_md_positions[0], input_positions)
    assert "minimization failed or was unstable" in caplog.text.lower()


def test_successful_minimization_seeds_md_with_minimized_coords(caplog):
    """A stable minimization seeds the MD with its final-frame coordinates."""
    atoms = ase_read(DATA_DIR / f"{STRUCTURE}.xyz")
    num_atoms = len(atoms)

    minimized_positions = atoms.get_positions() + 0.1
    minimization_state = _simulation_state(atoms, minimized_positions[np.newaxis, ...])
    md_state = _simulation_state(atoms, np.random.rand(2, num_atoms, 3))

    seen_md_positions = []

    def fake_run_simulation(atoms_arg, force_field, **kwargs):
        if kwargs.get("simulation_type") == "minimization":
            return minimization_state
        seen_md_positions.append(atoms_arg.get_positions().copy())
        return md_state

    with patch(
        "mlipaudit.utils.biomolecules.run_simulation",
        side_effect=fake_run_simulation,
    ):
        with caplog.at_level(logging.WARNING):
            results = list(
                iter_biomolecule_simulations(MagicMock(), DATA_DIR, RunMode.DEV)
            )

    assert results == [(STRUCTURE, md_state)]
    np.testing.assert_allclose(seen_md_positions[0], minimized_positions)
    assert "minimization failed or was unstable" not in caplog.text.lower()
