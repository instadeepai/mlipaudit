import mdtraj as md
import numpy as np
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, TypeAdapter


class Molecule(BaseModel):
    """Molecule class."""

    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    smiles: str
    pattern_atoms: list[int] | None
    charge: float


Molecules = TypeAdapter(dict[str, Molecule])


class MoleculeSimulationOutput(BaseModel):
    """Stores the simulation state for a molecule.

    Attributes:
        molecule_name: The name of the molecule.
        simulation_state: The simulation state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_name: str
    simulation_state: SimulationState


def create_mdtraj_from_simulation_state(
    simulation_state_positions: np.ndarray, atom_names: list[str]
) -> md.Trajectory:
    """Load a simulation state into an MDTraj trajectory.

    Args:
        simulation_state_positions: Atomic positions from a simulation state.
        atom_names: list of atom names.

    Returns:
        trajectory: MDTraj trajectory.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain)

    for name in atom_names:
        topology.add_atom(
            name=name, element=md.element.get_by_symbol(name), residue=residue
        )

    if simulation_state_positions.ndim == 2:
        positions = simulation_state_positions.reshape(1, -1, 3)
    else:
        positions = simulation_state_positions

    positions = positions / 10.0  # convert to nanometers
    trajectory = md.Trajectory(topology=topology, xyz=positions)

    return trajectory
