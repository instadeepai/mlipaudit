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

from dataclasses import dataclass

import numpy as np
import pydantic
import pytest

from mlipaudit.io_helpers import (
    dataclass_to_dict_with_arrays,
    dict_with_arrays_to_dataclass,
)


@dataclass
class LeafClass:
    """Example data class."""

    name: str
    value: int
    values_1: np.ndarray
    values_2: dict[str, np.ndarray]


@dataclass
class DataClass:
    """Example data class."""

    label: str
    value: float
    class_1: LeafClass
    classes_2: list[LeafClass]


class DataClassPydanticModel(pydantic.BaseModel):
    """Example pydantic model that acts as a dataclass."""

    label: str
    value: float
    class_1: LeafClass
    classes_2: list[LeafClass]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


@pytest.mark.parametrize("use_pydantic", [True, False])
def test_dataclass_can_be_turned_into_dictionary_with_numpy_arrays_separated_out(
    use_pydantic,
):
    """Tests the two helper functions that can turn a nested dataclass into two
    dictionaries, separating out the numpy arrays for separate saving.
    """
    _class = DataClassPydanticModel if use_pydantic else DataClass
    d_class = _class(
        label="l",
        value=12.34,
        class_1=LeafClass(
            "c1", 1, np.array([1.0, 2.0, 3.0]), {"a": np.array([4.0, 5.0, 6.0])}
        ),
        classes_2=[
            LeafClass(
                "c2a",
                2,
                np.array([[4.0, 5.0], [6.0, 7.0]]),
                {"b": np.array([[14.0, 15.0], [16.0, 17.0]])},
            ),
            LeafClass(
                "c2b",
                3,
                np.array([[40.0, 50.0], [60.0, 70.0]]),
                {
                    "b": np.array([[14.0, 15.0], [16.0, 17.0]]),
                    "c": np.array([1.0, 2.0, 3.0]),
                },
            ),
        ],
    )

    data, arrays = dataclass_to_dict_with_arrays(d_class)

    assert data["label"] == "l"
    assert data["class_1"]["value"] == 1
    assert data["class_1"]["values_2"]["a"] == "np_1"
    assert list(arrays.keys()) == [f"np_{i}" for i in range(7)]
    for i in range(7):
        assert isinstance(arrays[f"np_{i}"], np.ndarray)
    assert np.array_equal(arrays["np_1"], np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(arrays["np_5"], np.array([[14.0, 15.0], [16.0, 17.0]]))

    d_class_reconstructed = dict_with_arrays_to_dataclass(data, arrays, _class)

    assert isinstance(d_class_reconstructed, _class)
    assert d_class_reconstructed.label == "l"
    assert d_class_reconstructed.value == 12.34
    assert isinstance(d_class_reconstructed.class_1, LeafClass)
    assert isinstance(d_class_reconstructed.classes_2[0], LeafClass)
    assert sorted(d_class_reconstructed.classes_2[1].values_2.keys()) == ["b", "c"]
    assert np.array_equal(
        d_class_reconstructed.classes_2[1].values_2["b"],
        np.array([[14.0, 15.0], [16.0, 17.0]]),
    )
