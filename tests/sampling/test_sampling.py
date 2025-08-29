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
from unittest.mock import patch

import mdtraj as md
import numpy as np
import pytest
from ase.io import read as ase_read
from mlip.simulation import SimulationState

from mlipaudit.sampling.helpers import get_all_dihedrals_from_trajectory
from mlipaudit.sampling.sampling import (
    RESNAME_TO_BACKBONE_RESIDUE_TYPE,
    ResidueTypeBackbone,
    ResidueTypeSidechain,
    SamplingBenchmark,
)

DATA_DIR = Path(__file__).parent.parent / "data"
STRUCTURE_NAMES = ["ala_leu_glu_lys_sol"]


@pytest.fixture
def sampling_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> SamplingBenchmark:
    """Assembles a fully configured and isolated SamplingBenchmark instance.

    This fixture is parameterized to handle the `fast_dev_run` flag.

    Returns:
        An initialized SamplingBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)

    return SamplingBenchmark(
        force_field=mock_force_field,
        data_input_dir=DATA_DIR,
        fast_dev_run=is_fast_run,
    )


def test_get_all_dihedrals_from_trajectory():
    """Test the get_all_dihedrals_from_trajectory function."""
    traj_test = md.load_pdb(
        DATA_DIR / "sampling" / "pdb_reference_structures" / "ala_leu_glu_lys_sol.pdb"
    )

    dihedrals_data = get_all_dihedrals_from_trajectory(traj_test)
    assert len(dihedrals_data) == 4

    for _, value in dihedrals_data.items():
        assert "phi" in value.keys()
        assert "psi" in value.keys()

        assert value["phi"].shape == (1,)
        assert value["psi"].shape == (1,)


def test_data_loading(sampling_benchmark):
    """Test the data loading."""
    benchmark = sampling_benchmark
    backbone_reference_data, sidechain_reference_data = benchmark._reference_data()

    assert isinstance(backbone_reference_data, dict)
    assert isinstance(sidechain_reference_data, dict)

    assert all([
        x in backbone_reference_data.keys()
        for x in RESNAME_TO_BACKBONE_RESIDUE_TYPE.values()
    ])

    assert all([x in sidechain_reference_data.keys() for x in ["ASN", "PRO", "ARG"]])

    assert isinstance(backbone_reference_data["GLY"], ResidueTypeBackbone)
    assert isinstance(sidechain_reference_data["ASN"], ResidueTypeSidechain)

    assert len(backbone_reference_data["GLY"].phi) == 3
    assert len(backbone_reference_data["GLY"].psi) == 3

    assert len(sidechain_reference_data["ASN"].chi1) == 3
    assert len(sidechain_reference_data["ASN"].chi2) == 3
    assert sidechain_reference_data["ASN"].chi3 is None
    assert sidechain_reference_data["ASN"].chi4 is None
    assert sidechain_reference_data["ASN"].chi5 is None

    assert len(sidechain_reference_data["PRO"].chi1) == 3
    assert sidechain_reference_data["PRO"].chi2 is None
    assert sidechain_reference_data["PRO"].chi3 is None
    assert sidechain_reference_data["PRO"].chi4 is None
    assert sidechain_reference_data["PRO"].chi5 is None

    assert len(sidechain_reference_data["ARG"].chi1) == 3
    assert len(sidechain_reference_data["ARG"].chi2) == 3
    assert len(sidechain_reference_data["ARG"].chi3) == 3
    assert len(sidechain_reference_data["ARG"].chi4) == 3
    assert sidechain_reference_data["ARG"].chi5 is None


@pytest.mark.parametrize("sampling_benchmark", [True, False], indirect=True)
def test_sampling_benchmark_full_run_with_mock_engine(
    sampling_benchmark,
    mock_jaxmd_simulation_engine,
):
    """Test the sampling benchmark full run with mock engine."""
    benchmark = sampling_benchmark

    atoms = ase_read(
        DATA_DIR / "sampling" / "pdb_reference_structures" / "ala_leu_glu_lys_sol.pdb"
    )
    traj = np.array([atoms.positions] * 1)
    forces = np.zeros(shape=traj.shape)

    mock_engine = mock_jaxmd_simulation_engine(
        SimulationState(
            atomic_numbers=atoms.numbers,
            positions=traj,
            forces=forces,
            temperature=np.zeros(1),
        )
    )

    with patch(
        "mlipaudit.sampling.sampling.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        if benchmark.fast_dev_run:
            benchmark.run_model()
        else:
            with pytest.raises(FileNotFoundError):
                benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert mock_engine.run.call_count == 1

    benchmark.analyze()
