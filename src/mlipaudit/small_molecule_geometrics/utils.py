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
