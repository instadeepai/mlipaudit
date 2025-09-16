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
"""This script is designed to aid the construction of benchmarks,
by determining the required element types for each benchmark
based on the input data that is used. This was run once to
extract all the required element types for each benchmark, which
were then manually added to each benchmark under the attribute
`required_elements`. If adding a new benchmark class, we encourage
users to complete this script with a custom function calculating
the required element types for their new benchmark.
"""

import json
import os
from pathlib import Path

from ase import Atoms
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
from mlipaudit.tautomers.tautomers import TAUTOMERS_DATASET_FILENAME, TautomerPairs
from mlipaudit.water_radial_distribution.water_radial_distribution import WATERBOX_N500

DATA_LOCATION = "data"


def get_element_types_from_molecules(
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
    atom_element_types = set()
    if isinstance(molecules, dict):
        for pattern_name, molecule in molecules.items():
            atom_element_types.update(molecule.atom_symbols)
    elif isinstance(molecules, list):
        for molecule in molecules:
            atom_element_types.update(molecule.atom_symbols)
    return atom_element_types


def get_element_types_for_bld(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for bond length distribution.

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
    return get_element_types_from_molecules(molecules)


def get_element_types_for_cs(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for conformer selection.

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
    return get_element_types_from_molecules(conformers)


def get_element_types_for_ds(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for dihedral scan.

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
    return get_element_types_from_molecules(fragments)


def get_element_types_for_fs(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for folding stability.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    for structure_name in FS_STRUCTURE_NAMES:
        xyz_filename = structure_name + ".xyz"
        atoms = ase_read(Path(data_dir) / "folding_stability" / xyz_filename)
        atom_element_types.update(atoms.get_chemical_symbols())
    return atom_element_types


def get_element_types_for_nci(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for noncovalent interactions.

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
    return get_element_types_from_molecules(nci_atlas_data)


def get_element_types_for_r(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for reactivity.

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
    atom_element_types = set()
    for _, reaction in reactions.items():
        atom_element_types.update(reaction.products.atom_symbols)
        atom_element_types.update(reaction.reactants.atom_symbols)
        atom_element_types.update(reaction.transition_state.atom_symbols)
    return atom_element_types


def get_element_types_for_rp(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for ring planarity.

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
    return get_element_types_from_molecules(molecules)


def get_element_types_for_sampling(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for sampling.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    for system_name in SAMPLING_STRUCTURE_NAMES:
        xyz_filename = system_name + ".xyz"
        atoms = ase_read(
            Path(data_dir) / "sampling" / "starting_structures" / xyz_filename
        )
        atom_element_types.update(atoms.get_chemical_symbols())
    return atom_element_types


def get_element_types_for_scaling(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for ring planarity.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    for system_path in os.listdir(Path(data_dir) / "scaling"):
        xyz_filename = Path(system_path).stem + ".xyz"
        atoms = ase_read(Path(data_dir) / "scaling" / xyz_filename)
        atom_element_types.update(atoms.get_chemical_symbols())
    return atom_element_types


def get_element_types_for_smm(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for small molecule minimization.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    for dataset_filename in [
        QM9_NEUTRAL_FILENAME,
        QM9_CHARGED_FILENAME,
        OPENFF_NEUTRAL_FILENAME,
        OPENFF_CHARGED_FILENAME,
    ]:
        filepath = Path(data_dir) / "small_molecule_minimization" / dataset_filename
        with open(filepath, "r", encoding="utf-8") as f:
            molecules = SMMMolecules.validate_json(f.read())
            atom_element_types.update(get_element_types_from_molecules(molecules))
    return atom_element_types


def get_element_types_for_srd(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for solvent radial distribution.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    for system_name in BOX_CONFIG.keys():
        pdb_filename = system_name + "_eq.pdb"
        atoms = ase_read(Path(data_dir) / "solvent_radial_distribution" / pdb_filename)
        atom_element_types.update(atoms.get_chemical_symbols())
    return atom_element_types


def get_element_types_for_stability(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for stability.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    for structure_name in STABILITY_STRUCTURE_NAMES:
        xyz_filename = STABILITY_STRUCTURES[structure_name]["xyz"]
        atoms = ase_read(Path(data_dir) / "stability" / xyz_filename)
        atom_element_types.update(atoms.get_chemical_symbols())
    return atom_element_types


def get_element_types_for_t(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for tautomers.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atom_element_types = set()
    with open(
        Path(data_dir) / "tautomers" / TAUTOMERS_DATASET_FILENAME,
        mode="r",
        encoding="utf-8",
    ) as f:
        tautomers_dataset = TautomerPairs.validate_json(f.read())
        for _, tautomer_entry in tautomers_dataset.items():
            for i in range(2):
                atoms = Atoms(
                    symbols=tautomer_entry.atom_symbols[i],
                    positions=tautomer_entry.coordinates[i],
                )
                atom_element_types.update(atoms.get_chemical_symbols())
    return atom_element_types


def get_element_types_for_wrd(data_dir: os.PathLike | str) -> set[str]:
    """Fetch the element_types for water radial distribution.

    Args:
        data_dir: The directory containing the input data.

    Returns:
        The full set of atom symbols in the dataset.
    """
    atoms = ase_read(Path(data_dir) / "water_radial_distribution" / WATERBOX_N500)
    return set(atoms.get_chemical_symbols())


def main():
    """For each benchmark, preprocess the input files,
    compiling the element types contained in the input files,
    before saving the sets of element types to a json file.

    Note that the data will be fetched from the standard local
    data location, so these data files must be added manually
    beforehand, either manually or by running the benchmarks.
    """
    data_path = Path(__file__).parent.parent / DATA_LOCATION

    element_types_data = {
        "bld": list(get_element_types_for_bld(data_path)),
        "cs": list(get_element_types_for_cs(data_path)),
        "ds": list(get_element_types_for_ds(data_path)),
        "fs": list(get_element_types_for_fs(data_path)),
        "nci": list(get_element_types_for_nci(data_path)),
        "r": list(get_element_types_for_r(data_path)),
        "rp": list(get_element_types_for_rp(data_path)),
        "smm": list(get_element_types_for_smm(data_path)),
        "srd": list(get_element_types_for_srd(data_path)),
        "sampling": list(get_element_types_for_sampling(data_path)),
        "scaling": list(get_element_types_for_scaling(data_path)),
        "stability": list(get_element_types_for_stability(data_path)),
        "t": list(get_element_types_for_t(data_path)),
        "wrd": list(get_element_types_for_wrd(data_path)),
    }

    output_file = "element_types_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(element_types_data, f, indent=4)

    print(f"Data successfully saved to {output_file}")


if __name__ == "__main__":
    main()
