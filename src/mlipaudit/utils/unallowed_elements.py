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

from ase.data import chemical_symbols
from mlip.models import ForceField


def skip_unallowed_elements(
    force_field: ForceField,
    structure_tuples: list[tuple[str, list[str]]],
) -> list[str]:
    """Get a list of structure identifiers that contain unallowed elements.

    Args:
        force_field: The force field to use.
        structure_tuples: A list of tuples, where each tuple contains a structure
            identifier and a list of atom symbols.

    Returns:
        A list of structure identifiers that contain unallowed elements.
    """
    allowed_atomic_numbers = force_field.allowed_atomic_numbers
    allowed_symbols = set(chemical_symbols[z] for z in allowed_atomic_numbers)

    structures_to_skip = []

    for structure_id, atom_symbols_list in structure_tuples:
        if not set(atom_symbols_list).issubset(allowed_symbols):
            structures_to_skip.append(structure_id)

    return structures_to_skip
