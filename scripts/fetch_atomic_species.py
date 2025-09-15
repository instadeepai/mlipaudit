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
import json
import os
from pathlib import Path

from ase.io import read as ase_read
from pydantic import BaseModel

from mlipaudit.bond_length_distribution.bond_length_distribution import (
    BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME,
)
from mlipaudit.bond_length_distribution.bond_length_distribution import (
    Molecules as BLDMolecules,
)
from mlipaudit.conformer_selection.conformer_selection import (
    WIGGLE_DATASET_FILENAME,
    Conformers,
)
from mlipaudit.dihedral_scan.dihedral_scan import TORSIONNET_DATASET_FILENAME, Fragments
from mlipaudit.folding_stability.folding_stability import (
    STRUCTURE_NAMES as FS_STRUCTURE_NAMES,
)
from mlipaudit.noncovalent_interactions.noncovalent_interactions import (
    NCI_ATLAS_FILENAME,
    Systems,
)
from mlipaudit.reactivity.reactivity import GRAMBOW_DATASET_FILENAME, Reactions
from mlipaudit.ring_planarity.ring_planarity import RING_PLANARITY_DATASET
from mlipaudit.ring_planarity.ring_planarity import Molecules as RPMolecules
from mlipaudit.sampling.sampling import STRUCTURE_NAMES as SAMPLING_STRUCTURE_NAMES
from mlipaudit.small_molecule_minimization.small_molecule_minimization import (
    OPENFF_CHARGED_FILENAME,
    OPENFF_NEUTRAL_FILENAME,
    QM9_CHARGED_FILENAME,
    QM9_NEUTRAL_FILENAME,
)
from mlipaudit.small_molecule_minimization.small_molecule_minimization import (
    Molecules as SMMMolecules,
)
from mlipaudit.solvent_radial_distribution.solvent_radial_distribution import BOX_CONFIG
from mlipaudit.stability.stability import STRUCTURE_NAMES as STABILITY_STRUCTURE_NAMES
from mlipaudit.stability.stability import STRUCTURES as STABILITY_STRUCTURES


def get_species_from_molecules(
    molecules: dict[str, BaseModel] | list[BaseModel],
) -> set[str]:
    """Given a dictionary or list of Molecules, containing
    the field `atom_symbols`, fetch the full set of atom symbols
    in the dataset.

    Args:
        molecules: The dictionary or list of Molecule base classes.

    Returns:
        A set of the atom symbols in the dataset.
    """
    atom_species = set()
    if isinstance(molecules, dict):
        for pattern_name, molecule in molecules.items():
            atom_species.update(molecule.atom_symbols)
    elif isinstance(molecules, list):
        for molecule in molecules:
            atom_species.update(molecule.atom_symbols)
    return atom_species


def get_species_for_bld(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for bond length distribution.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir)
        / "bond_length_distribution"
        / BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME,
        mode="r",
        encoding="utf-8",
    ) as f:
        molecules = BLDMolecules.validate_json(f.read())
    return get_species_from_molecules(molecules)


def get_species_for_cs(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for conformer selection.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir) / "conformer_selection" / WIGGLE_DATASET_FILENAME,
        mode="r",
        encoding="utf-8",
    ) as f:
        conformers = Conformers.validate_json(f.read())
    return get_species_from_molecules(conformers)


def get_species_for_ds(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for dihedral scan.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir) / "dihedral_scan" / TORSIONNET_DATASET_FILENAME,
        mode="r",
        encoding="utf-8",
    ) as f:
        fragments = Fragments.validate_json(f.read())
    return get_species_from_molecules(fragments)


def get_species_for_fs(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for folding stability.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_species = set()
    for structure_name in FS_STRUCTURE_NAMES:
        xyz_filename = structure_name + ".xyz"
        atoms = ase_read(Path(data_dir) / "folding_stability" / xyz_filename)
        atom_species.update(atoms.get_chemical_symbols())
    return atom_species


def get_species_for_nci(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for noncovalent interactions.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir) / "noncovalent_interactions" / NCI_ATLAS_FILENAME,
        "r",
        encoding="utf-8",
    ) as f:
        nci_atlas_data = Systems.validate_json(f.read())
    return get_species_from_molecules(nci_atlas_data)


def get_species_for_r(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for reactivity.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir) / "reactivity" / GRAMBOW_DATASET_FILENAME,
        "r",
        encoding="utf-8",
    ) as f:
        reactions = Reactions.validate_json(f.read())
    atom_species = set()
    for _, reaction in reactions.items():
        atom_species.update(reaction.products.atom_symbols)
        atom_species.update(reaction.reactants.atom_symbols)
        atom_species.update(reaction.transition_state.atom_symbols)
    return atom_species


def get_species_for_rp(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for ring planarity.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir) / "ring_planarity" / RING_PLANARITY_DATASET,
        mode="r",
        encoding="utf-8",
    ) as f:
        molecules = RPMolecules.validate_json(f.read())
    return get_species_from_molecules(molecules)


def get_species_for_sampling(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for sampling.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_species = set()
    for system_name in SAMPLING_STRUCTURE_NAMES:
        xyz_filename = system_name + ".xyz"
        atoms = ase_read(
            Path(data_dir) / "sampling" / "starting_structures" / xyz_filename
        )
        atom_species.update(atoms.get_chemical_symbols())
    return atom_species


def get_species_for_scaling(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for ring planarity.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_species = set()
    for system_path in os.listdir(Path(data_dir) / "scaling"):
        xyz_filename = Path(system_path).stem + ".xyz"
        atoms = ase_read(Path(data_dir) / "scaling" / xyz_filename)
        atom_species.update(atoms.get_chemical_symbols())
    return atom_species


def get_species_for_smm(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for small molecule minimization.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_species = set()
    for dataset_filename in [
        QM9_NEUTRAL_FILENAME,
        QM9_CHARGED_FILENAME,
        OPENFF_NEUTRAL_FILENAME,
        OPENFF_CHARGED_FILENAME,
    ]:
        filepath = Path(data_dir) / "small_molecule_minimization" / dataset_filename
        with open(filepath, "r", encoding="utf-8") as f:
            molecules = SMMMolecules.validate_json(f.read())
            atom_species.update(get_species_from_molecules(molecules))
    return atom_species


def get_species_for_srd(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for solvent radial distribution.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_species = set()
    for system_name in BOX_CONFIG.keys():
        pdb_filename = system_name + "_eq.pdb"
        atoms = ase_read(Path(data_dir) / "solvent_radial_distribution" / pdb_filename)
        atom_species.update(atoms.get_chemical_symbols())
    return atom_species


def get_species_for_stability(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for stability.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_species = set()
    for structure_name in STABILITY_STRUCTURE_NAMES:
        xyz_filename = STABILITY_STRUCTURES[structure_name]["xyz"]
        atoms = ase_read(Path(data_dir) / "stability" / xyz_filename)
        atom_species.update(atoms.get_chemical_symbols())
    return atom_species


def get_species_for_t(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the species for tautomers.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    with open(
        Path(data_dir) / "tautomers" / RING_PLANARITY_DATASET,
        mode="r",
        encoding="utf-8",
    ) as f:
        molecules = RPMolecules.validate_json(f.read())
    return get_species_from_molecules(molecules)


def main():
    """For each benchmark, preprocess the input files,
    compiling the atomic species contained in the input files,
    before saving the sets of atomic species to a json file.

    Note that the data will be fetched from the standard local
    data location, so these data files must be added manually
    beforehand, either manually or by running the benchmarks.
    """
    data_path = Path(__file__).parent.parent / "data"

    species_data = {
        "bld": list(get_species_for_bld(data_path)),
        "cs": list(get_species_for_cs(data_path)),
        "ds": list(get_species_for_ds(data_path)),
        "fs": list(get_species_for_fs(data_path)),
        "nci": list(get_species_for_nci(data_path)),
        "r": list(get_species_for_r(data_path)),
        "rp": list(get_species_for_rp(data_path)),
        "smm": list(get_species_for_smm(data_path)),
        "srd": list(get_species_for_srd(data_path)),
        "sampling": list(get_species_for_sampling(data_path)),
        "scaling": list(get_species_for_scaling(data_path)),
        "stability": list(get_species_for_stability(data_path)),
    }

    output_file = "species_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(species_data, f, indent=4)

    print(f"Data successfully saved to {output_file}")


if __name__ == "__main__":
    main()
