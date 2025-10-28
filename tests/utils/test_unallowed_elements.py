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


import re

import pytest

from mlipaudit.exceptions import ChemicalElementsMissingError
from mlipaudit.utils import skip_unallowed_elements


def test_allowed_elements_are_not_skipped(mock_force_field):
    """Tests the skip_unallowed_elements function."""
    mock_force_field.allowed_atomic_numbers = list(range(1, 92))

    structure_tuples = [
        ("mol_1", ["H", "C", "N", "O"]),
        ("mol_2", ["H", "H", "H", "C", "H", "O"]),
        ("mol_3", ["H", "H", "H", "C", "N", "H", "H"]),
        ("mol_4", ["He", "C", "H", "H", "H", "H"]),
        ("mol_5", ["He", "B", "Si", "Ge", "As", "Na", "Cl"]),
    ]

    assert skip_unallowed_elements(mock_force_field, structure_tuples) == []


def test_unallowed_elements_are_skipped(mock_force_field):
    """Tests the skip_unallowed_elements function."""
    mock_force_field.allowed_atomic_numbers = [1, 6, 7, 8]

    structure_tuples = [
        ("mol_1", ["H", "C", "N", "O"]),
        ("mol_2", ["He", "H", "H", "C", "H", "O"]),
        ("mol_3", ["F", "F", "F", "C", "N", "H", "H"]),
    ]

    assert skip_unallowed_elements(mock_force_field, structure_tuples) == [
        "mol_2",
        "mol_3",
    ]


def test_check_can_run_model(mock_force_field, dummy_benchmark_1_class):
    """Tests the check_can_run_model function."""
    assert dummy_benchmark_1_class.check_can_run_model(mock_force_field) is True

    mock_force_field.allowed_atomic_numbers = {"H"}

    assert dummy_benchmark_1_class.check_can_run_model(mock_force_field) is False

    expected_message = "The following element types are missing: {'O'}"
    with pytest.raises(ChemicalElementsMissingError, match=re.escape(expected_message)):
        dummy_benchmark_1_class(mock_force_field)
