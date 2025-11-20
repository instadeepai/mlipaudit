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

from mlipaudit.benchmarks.sampling.helpers import (
    calculate_distribution_hellinger_distance,
    calculate_distribution_kl_divergence,
    calculate_distribution_rmsd,
    calculate_multidimensional_distribution,
    get_all_dihedrals_from_trajectory,
    identify_outlier_data_points,
)
from mlipaudit.benchmarks.sampling.sampling import (
    RESNAME_TO_BACKBONE_RESIDUE_TYPE,
    ResidueTypeBackbone,
    ResidueTypeSidechain,
    SamplingBenchmark,
    SamplingResult,
    SamplingSystemResult,
)
from mlipaudit.run_mode import RunMode

DATA_DIR = Path(__file__).parent.parent / "data"
STRUCTURE_NAMES = ["chignolin_1uao_xray"]


@pytest.fixture
def sampling_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> SamplingBenchmark:
    """Assembles a fully configured and isolated SamplingBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized SamplingBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return SamplingBenchmark(
        force_field=mock_force_field,
        data_input_dir=DATA_DIR,
        run_mode=run_mode,
    )


def test_get_all_dihedrals_from_trajectory():
    """Test the get_all_dihedrals_from_trajectory function."""
    traj_test = md.load_pdb(
        DATA_DIR / "sampling" / "pdb_reference_structures" / "chignolin_1uao_xray.pdb"
    )

    dihedrals_data = get_all_dihedrals_from_trajectory(traj_test)
    assert len(dihedrals_data) == 8

    for _, value in dihedrals_data.items():
        assert "phi" in value.keys()
        assert "psi" in value.keys()

        assert value["phi"].shape == (1,)
        assert value["psi"].shape == (1,)


def test_calculate_multidimensional_distribution():
    """Test the calculate_multidimensional_distribution function."""
    points = np.array([
        [-180.0, -180.0],
        [-180.0, -180.0],
        [-180.0, -180.0],
        [180.0, 180.0],
        [180.0, 180.0],
        [180.0, 180.0],
    ])
    hist, _ = calculate_multidimensional_distribution(points, bins=2)

    assert hist.shape == (2, 2)
    assert np.sum(hist) == pytest.approx(1.0)

    assert hist[0, 0] == pytest.approx(0.5)
    assert hist[0, 1] == pytest.approx(0.0)
    assert hist[1, 0] == pytest.approx(0.0)
    assert hist[1, 1] == pytest.approx(0.5)


def test_calculate_distribution_rmsd():
    """Test the calculate_distribution_rmsd function."""
    hist1 = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.0],
    ])
    hist2 = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    hist2_unnormed = np.array([
        [2.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    assert calculate_distribution_rmsd(hist1, hist1) == pytest.approx(0.0)
    assert calculate_distribution_rmsd(hist1, hist2) == pytest.approx(0.5)
    assert calculate_distribution_rmsd(hist1, hist2_unnormed) == pytest.approx(0.5)


def test_calculate_distribution_kl_divergence():
    """Test the calculate_distribution_kl_divergence function."""
    hist1 = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.0],
    ])
    hist2 = np.array([
        [0.0, 0.1],
        [0.2, 0.2],
        [0.0, 0.0],
    ])
    hist2_unnormed = np.array([
        [0.0, 0.2],
        [0.4, 0.4],
        [0.0, 0.0],
    ])

    assert calculate_distribution_kl_divergence(hist1, hist1) == pytest.approx(0.0)
    assert calculate_distribution_kl_divergence(hist1, hist2) == pytest.approx(
        0.2231435513142097
    )
    assert calculate_distribution_kl_divergence(hist1, hist2_unnormed) == pytest.approx(
        0.2231435513142097
    )


def test_calculate_distribution_hellinger_distance():
    """Test the calculate_distribution_hellinger_distance function."""
    hist1 = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.0],
    ])

    hist2 = np.array([
        [0.5, 0.5],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    hist3 = np.array([
        [0.2, 0.2],
        [0.3, 0.3],
        [0.0, 0.0],
    ])

    assert calculate_distribution_hellinger_distance(hist1, hist1) == pytest.approx(0.0)
    assert calculate_distribution_hellinger_distance(hist1, hist2) == pytest.approx(1.0)
    assert calculate_distribution_hellinger_distance(hist1, hist3) == pytest.approx(
        0.4747666066168898
    )


def test_identify_outlier_data_points():
    """Test the identify_outlier_data_points function."""
    sampled_dihedrals = np.array([
        [-180.0, -180.0],
        [1.0, 0.0],
        [90.0, 90.0],
    ])
    reference_dihedrals = np.array([
        [180.0, 180.0],
        [0.0, 0.0],
    ])

    outliers = identify_outlier_data_points(sampled_dihedrals, reference_dihedrals)

    assert outliers == [False, False, True]


def test_data_loading(sampling_benchmark):
    """Test the loading of the reference data."""
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
        DATA_DIR / "sampling" / "pdb_reference_structures" / "chignolin_1uao_xray.pdb"
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
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        if benchmark.run_mode == RunMode.DEV:
            benchmark.run_model()
        else:
            with pytest.raises(FileNotFoundError):
                benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert mock_engine.run.call_count == 1

    with patch(
        "mlipaudit.benchmarks.sampling.sampling.SamplingBenchmark."
        "_assert_structure_names_in_model_output"
    ):
        results = benchmark.analyze()

    assert isinstance(results, SamplingResult)
    assert isinstance(results.systems[0], SamplingSystemResult)

    assert len(results.systems) == 1
    assert len(results.exploded_systems) == 0

    assert len(results.rmsd_backbone_dihedrals) == 7
    assert len(results.hellinger_distance_backbone_dihedrals) == 7
    assert len(results.rmsd_sidechain_dihedrals) == 6
    assert len(results.hellinger_distance_sidechain_dihedrals) == 6

    allowed_bb = ["THR", "GLU", "GLY", "TRP", "PRO", "ASP", "TYR"]
    allowed_sc = ["THR", "GLU", "PRO", "TRP", "ASP", "TYR"]

    assert all(x in allowed_bb for x in results.rmsd_backbone_dihedrals.keys())
    assert all(
        x in allowed_bb for x in results.hellinger_distance_backbone_dihedrals.keys()
    )
    assert all(x in allowed_sc for x in results.rmsd_sidechain_dihedrals.keys())
    assert all(
        x in allowed_sc for x in results.hellinger_distance_sidechain_dihedrals.keys()
    )
