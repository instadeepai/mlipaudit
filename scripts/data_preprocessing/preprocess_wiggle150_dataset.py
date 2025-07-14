"""Fetch and process wiggle 150 dataset of highly strained conformers.

This script will download the wiggle 150 dataset from the supplementary material of its
ChemArxiv publication (DOI 10.26434/chemrxiv-2025-4mbsk-v3). The data is processed into
a json file containing the conformer name, associated dft energy, atom names and
coordinates.

Optionally, the script will also make separate xyz files of the conformers for
debugging and dataset visualization.
"""

import logging
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read
from pydantic import TypeAdapter
from utils import download_file

from mlipaudit.small_molecule_conformer_selection.conformer_selection import Conformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = (
    Path(__file__).parent / ".." / ".." / "data" / "small_molecule_conformer_selection"
).resolve()

DATA_URL = "https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/677c52a981d2151a02209701/original/si-coords-xyz.xyz"

INPUT_XYZ_FILENAME = "wiggle150.xyz"
OUTPUT_FILENAME = "wiggle150_dataset.json"


def get_conf_name(
    atoms: Atoms, conformer_names: list[str] = ["ado", "bpn", "efa"]
) -> str:
    """Given an Atoms object, return the conformer name.

    Args:
        atoms: Atoms object.
        conformer_names: List of conformer names.

    Returns:
        The conformer name of the Atoms object.

    Raises:
        ValueError: If the conformer does not have
            a valid name.
    """
    for key in atoms.info.keys():
        if key[:3] in conformer_names:
            return key
    raise ValueError(f"Atom {atoms} does not have a valid conformer name.")


def get_dft_energy(atoms: Atoms) -> float:
    """Given an Atoms object, return the dft energy.

    Args:
        atoms: Atoms object.

    Returns:
        The dft energy of the Atoms object.

    Raises:
        ValueError: If the conformer does not have
            a valid energy.
    """
    for key in atoms.info.keys():
        try:
            # Attempt to convert the key to a float
            return float(key)
        except ValueError:
            continue
    raise ValueError(f"Atom {atoms} does not have a valid dft energy.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Process wiggle 150 dataset by downloading as one xyz file and "
            "processing into benchmark database."
        )
    )
    parser.add_argument(
        "--make-xyz",
        action="store_true",
        help="Make separate xyz files of the conformers.",
    )

    args = parser.parse_args()

    if args.make_xyz:
        XYZ_DIR = DATA_DIR / "xyz"
        XYZ_DIR.mkdir(parents=True, exist_ok=True)

    download_file(DATA_URL, str(DATA_DIR / INPUT_XYZ_FILENAME))

    dataset: list[dict] = [
        {
            "molecule_name": "ado",
            "dft_energy_profile": [],
            "atom_symbols": [],
            "conformer_coordinates": [],
        },
        {
            "molecule_name": "bpn",
            "dft_energy_profile": [],
            "atom_symbols": [],
            "conformer_coordinates": [],
        },
        {
            "molecule_name": "efa",
            "dft_energy_profile": [],
            "atom_symbols": [],
            "conformer_coordinates": [],
        },
    ]

    atoms_list = ase_read(str(DATA_DIR / INPUT_XYZ_FILENAME), index=":")

    molecule_names = ["ado", "bpn", "efa"]

    for atoms in atoms_list:
        conf_name = get_conf_name(atoms).split("_")[0]
        dft_energy = get_dft_energy(atoms)
        atoms_symbols = atoms.get_chemical_symbols()
        coords = atoms.get_positions()

        for conformer_set in dataset:
            if conf_name == conformer_set["molecule_name"]:
                if len(conformer_set["atom_symbols"]) == 0:
                    conformer_set["atom_symbols"] = atoms_symbols
                elif atoms_symbols != conformer_set["atom_symbols"]:
                    raise ValueError(
                        f"Atom mismatch for {conf_name} and"
                        f" {conformer_set['molecule_name']}"
                    )

                conformer_set["dft_energy_profile"].append(dft_energy)
                conformer_set["conformer_coordinates"].append(coords)

    list_of_molecules_adapter = TypeAdapter(list[Conformer])
    conf_list = []
    for conformer in dataset:
        conf_list.append(Conformer.model_validate(conformer))

    json_output_bytes = list_of_molecules_adapter.dump_json(conf_list, indent=2)
    with open(DATA_DIR / OUTPUT_FILENAME, "wb") as fb:
        fb.write(json_output_bytes)
        logging.info("Wrote dataset to disk at path %s", DATA_DIR / OUTPUT_FILENAME)

    if args.make_xyz:
        for atoms in atoms_list:
            conf_name = get_conf_name(atoms).split("_")[0]
            dft_energy = get_dft_energy(atoms)
            atoms_symbols = atoms.get_chemical_symbols()
            coords = atoms.get_positions()
            xyz_file = XYZ_DIR / f"{conf_name}.xyz"
            with open(xyz_file, "w", encoding="utf-8") as f:
                f.write(f"{len(atoms)}\n")
                f.write(f"{conf_name}\n")
                for atom, coord in zip(atoms, coords):
                    f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        logging.info("Wrote xyz files to disk at path %s", XYZ_DIR)
