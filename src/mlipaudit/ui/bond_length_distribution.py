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


from typing import Callable, TypeAlias

import streamlit as st

from mlipaudit.bond_length_distribution.bond_length_distribution import (
    BondLengthDistributionResult,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, BondLengthDistributionResult
]


def bond_length_distribution_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for bond length distribution.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Bond length distribution")
    st.sidebar.markdown("# Bond length distribution")

    st.markdown(
        "The benchmark runs short simulations of small molecules to check whether the "
        "correct bond lengths of typical bonds found in organic small molecules are "
        "preserved. This is an important test to see if the MLIP respects basic "
        "chemistry throughout simulations. For every bond type, the benchmark runs"
        " a short simulation of a test molecule and the bond length of that bond"
        " type is recorded. The key metric is the average deviation of the bond"
        " length throughout the simulation. This value should not exceed 0.025 Å."
    )
