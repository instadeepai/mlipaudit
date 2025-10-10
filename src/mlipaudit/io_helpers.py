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

import inspect
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, Protocol, Type, TypeVar, get_args

import jax.numpy as jnp
import numpy as np
import pydantic

T = TypeVar("T")


# Custom version of issubclass() that works with pydantic, too
def _is_subclass(cls: Any, parent_cls: Any) -> bool:
    try:
        return parent_cls in inspect.getmro(cls)
    except AttributeError:
        return False  # not a class


def _is_dataclass_or_pydantic_model(obj: Any) -> bool:
    return (
        is_dataclass(obj)
        or isinstance(obj, pydantic.BaseModel)
        or _is_subclass(obj, pydantic.BaseModel)
    )


class DataclassType(Protocol):
    """Something we can use in the type annotations to represent a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


def dataclass_to_dict_with_arrays(
    d_class: DataclassType | pydantic.BaseModel,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Converts a dataclass (possibly nested) into two separate dictionaries.

    Replace numpy arrays with string keys ("np_0", "np_1", ...),
    and collect arrays in a second dictionary for separate saving.

    Args:
        d_class: The dataclass to convert.

    Returns:
        The resulting dictionary of the dataclass and the separated arrays.
    """
    arrays = {}
    counter = [0]  # mutable counter

    def recurse(value):
        if _is_dataclass_or_pydantic_model(value):
            if is_dataclass(value):
                return {f.name: recurse(getattr(value, f.name)) for f in fields(value)}
            return {
                f: recurse(getattr(value, f)) for f in type(value).model_fields.keys()
            }
        elif isinstance(value, dict):
            return {k: recurse(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [recurse(v) for v in value]
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            key = f"np_{counter[0]}"
            arrays[key] = value
            counter[0] += 1
            return key
        else:
            return value

    return recurse(d_class), arrays


def dict_with_arrays_to_dataclass(
    data_with_array_placeholders: dict[str, Any],
    arrays: dict[str, np.ndarray],
    cls: Type[T],
) -> T:
    """Reconstructs a dataclass of type `cls` from a dictionary with numpy array keys
    and a dictionary of arrays.

    Args:
        data_with_array_placeholders: The dictionary that does not contain the arrays.
        arrays: The separate arrays dictionary.
        cls: The class (dataclass, pydantic model) that the dictionary
             should be loaded into.

    Returns:
        The instantiated class with the full data.
    """

    def recurse(value, expected_type=None):
        if (
            isinstance(value, dict)
            and expected_type
            and _is_dataclass_or_pydantic_model(expected_type)
        ):
            # Rebuild dataclass or pydantic BaseModel
            if is_dataclass(expected_type):
                field_types = {f.name: f.type for f in fields(expected_type)}
            else:
                field_types = {
                    name: field.annotation
                    for name, field in expected_type.model_fields.items()
                }
            return expected_type(**{
                k: recurse(v, field_types.get(k)) for k, v in value.items()
            })
        elif isinstance(value, dict):
            return {k: recurse(v) for k, v in value.items()}
        elif isinstance(value, list):
            item_type = get_args(expected_type)[0]
            exp_type = item_type if _is_dataclass_or_pydantic_model(item_type) else None
            return [recurse(v, exp_type) for v in value]
        elif isinstance(value, str) and value.startswith("np_"):
            # Replace with numpy array
            return arrays[value]
        else:
            return value

    return recurse(data_with_array_placeholders, cls)
