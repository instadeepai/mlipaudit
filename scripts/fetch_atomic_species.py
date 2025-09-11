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
import os
from pathlib import Path
from mlipaudit.bond_length_distribution.bond_length_distribution import Molecules as BLDMolecules
from mlipaudit.bond_length_distribution.bond_length_distribution import BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME
from mlipaudit.conformer_selection.conformer_selection import Conformers
from mlipaudit.conformer_selection.conformer_selection import WIGGLE_DATASET_FILENAME
from mlipaudit.dihedral_scan.dihedral_scan import Fragments
from mlipaudit.dihedral_scan.dihedral_scan import TORSIONNET_DATASET_FILENAME
from mlipaudit.folding_stability.folding_stability import STRUCTURE_NAMES as FS_STRUCTURE_NAMES
from ase.io import read as ase_read
from pydantic import BaseModel

from mlipaudit.noncovalent_interactions.noncovalent_interactions import NCI_ATLAS_FILENAME, Systems


def get_species_from_molecules(molecules: dict[str, BaseModel] | list[BaseModel]) -> set[str]:
    atom_species = set()
    if isinstance(molecules, dict):
        for pattern_name, molecule in molecules.items():
            atom_species.update(molecule.atom_symbols)
    elif isinstance(molecules, list):
        for molecule in molecules:
            atom_species.update(molecule.atom_symbols)
    return atom_species

def get_species_for_bld(dir: os.PathLike | str) -> set[str]:
    with open(
            dir / "bond_length_distribution" / BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
    ) as f:
        molecules = BLDMolecules.validate_json(f.read())
        return get_species_from_molecules(molecules)

def get_species_for_cs(dir: os.PathLike | str) -> set[str]:
    with open(
        dir / "conformer_selection" / WIGGLE_DATASET_FILENAME,
        mode="r",
        encoding="utf-8",
    ) as f:
        conformers = Conformers.validate_json(f.read())
        return get_species_from_molecules(conformers)

def get_species_for_ds(dir: os.PathLike | str) -> set[str]:
    with open(
            dir / "dihedral_scan" / TORSIONNET_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
    ) as f:
        fragments = Fragments.validate_json(f.read())
        return get_species_from_molecules(fragments)

def get_species_for_fs(dir: os.PathLike | str) -> set[str]:
    atom_species = set()
    for structure_name in FS_STRUCTURE_NAMES:
        xyz_filename = structure_name + ".xyz"
        atoms = ase_read(dir / "folding_stability" / xyz_filename)
        atom_species.update(atoms.get_chemical_symbols())
    return atom_species


def get_species_for_nci(dir: os.PathLike | str) -> set[str]:
    with open(
            dir / "noncovalent_interactions" / NCI_ATLAS_FILENAME,
            "r",
            encoding="utf-8",
    ) as f:
        nci_atlas_data = Systems.validate_json(f.read())
        return get_species_from_molecules(nci_atlas_data)


def main():
    """For each benchmark, preprocess the input files,
    compiling the atomic species contained in the input files."""
    print(get_species_for_bld(Path(__file__).parent.parent / "data"))
    print(get_species_for_cs(Path(__file__).parent.parent / "data"))
    print(get_species_for_ds(Path(__file__).parent.parent / "data"))
    print(get_species_for_fs(Path(__file__).parent.parent / "data"))
    print(get_species_for_nci(Path(__file__).parent.parent / "data"))

if __name__ == "__main__":
    main()