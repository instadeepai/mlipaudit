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
import pytest
from mlip.models import Esen, Mace, Nequip, Visnet

from mlipaudit.benchmarks_cli import _model_class_from_name


@pytest.mark.parametrize(
    "model_name,expected_class",
    [
        ("my_visnet_model.zip", Visnet),
        ("mace.zip", Mace),
        ("model_nequip.zip", Nequip),
        ("esen_model.zip", Esen),
    ],
)
def test_model_class_is_inferred_from_zip_name(model_name, expected_class):
    """The model architecture is inferred from a substring of the zip file name."""
    assert _model_class_from_name(model_name) is expected_class


def test_model_class_raises_for_unknown_name():
    """An unrecognized model name raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        _model_class_from_name("some_unknown_model.zip")
