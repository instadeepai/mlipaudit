"""Common fixtures across several tests."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from jraph import GraphsTuple
from mlip.data.dataset_info import DatasetInfo
from mlip.models import ForceField, Mace
from mlip.simulation.utils import create_graph_from_atoms

CUTOFF_ANGSTROM = 3.0


DATA_DIR = Path(__file__).parent / "data"
XYZ_FILE_PATH = DATA_DIR / "dimethyl_sulfoxide.xyz"
SAMPLE_ROTAMER_FILE = DATA_DIR / "sample_rotamer_file.data"


@pytest.fixture(scope="session")
def setup_system() -> tuple[list[Atoms], GraphsTuple, DatasetInfo]:
    """Set up atoms, graph and dataset info.

    Returns:
        A tuple of atoms, graph and dataset info.
    """
    atoms = ase_read(XYZ_FILE_PATH)
    positions = atoms.get_positions()

    senders, receivers = [], []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i != j and np.linalg.norm(positions[i] - positions[j]) < CUTOFF_ANGSTROM:
                senders.append(i)
                receivers.append(j)
    senders, receivers = np.asarray(senders), np.asarray(receivers)

    def displacement_fun(vec1, vec2):
        return vec1 - vec2

    allowed_z_numbers = set(range(1, 93))  # Allow all atoms
    graph = create_graph_from_atoms(
        atoms,
        senders,
        receivers,
        displacement_fun,
        allowed_atomic_numbers=allowed_z_numbers,
    )
    dataset_info = DatasetInfo(
        atomic_energies_map=dict.fromkeys(allowed_z_numbers, 0),
        avg_num_neighbors=6.8,
        avg_r_min_angstrom=None,
        cutoff_distance_angstrom=CUTOFF_ANGSTROM,
        scaling_mean=0.0,
        scaling_stdev=1.0,
    )

    return atoms, graph, dataset_info


@pytest.fixture(scope="session")
def setup_system_and_force_field(
    setup_system,
) -> tuple[list[Atoms], GraphsTuple, ForceField]:
    """Given a system, setup a mace ff.

    Returns:
        A tuple of atoms, graph and force field.
    """
    atoms, graph, dataset_info = setup_system

    mace_kwargs = {
        "num_layers": 2,
        "num_bessel": 8,
        "radial_envelope": "polynomial_envelope",
        "activation": "silu",
        "num_channels": 4,
        "readout_irreps": ("4x0e", "0e"),
        "correlation": 2,
        "node_symmetry": 2,
        "l_max": 2,
        "symmetric_tensor_product_basis": True,
    }

    mace_model = Mace(Mace.Config(**mace_kwargs), dataset_info)
    mace_ff = ForceField.from_mlip_network(
        mace_model,
        seed=42,
        predict_stress=False,
    )

    return atoms, graph, mace_ff
