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


import statistics
from typing import Callable, TypeAlias

import pandas as pd
import streamlit as st

from mlipaudit.small_molecule_geometrics.ring_planarity import RingPlanarityResult

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, RingPlanarityResult]


def ring_planarity_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for ring planarity.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Aromatic ring planarity")
    st.sidebar.markdown("# Aromatic ring planarity")

    st.markdown(
        "The benchmark runs short simulations of aromatic systems to check whether the "
        "ring planarity is preserved. This is a test if the MLIP respects aromaticity "
        "throughout simulations. The benchmarks runs a small set of aromatic systems "
        "and calculated the deviation of the ring atoms from the ideal plane. While "
        "some deviation is expected, especially for larger systems like indole, the "
        "average deviation throughout the simulation should be below 0.3 Å for all "
        "test systems."
    )

    st.markdown(
        "For more information, see the [docs](https://mlipaudit-dot-int-research-tpu.uc.r.appspot.com/benchmarks/small-molecules/ring_planarity.html)."
    )

    # Download data and get model names
    if "cached_data" not in st.session_state:
        st.session_state.cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.cached_data

    # unique_model_names = list(set(data.keys()))
    # model_select = st.sidebar.multiselect(
    #     "Select model(s)", unique_model_names, default=unique_model_names
    # )
    # selected_models = model_select if model_select else unique_model_names

    deviation_data = [
        {
            "Model name": model_name,
            "Average deviations": [
                mol_result.avg_deviation for mol_result in result.molecule_results
            ],
            "Average deviation": statistics.mean(
                mol_result.avg_deviation for mol_result in result.molecule_results
            ),
        }
        for model_name, result in data.items()
    ]

    df_deviation = pd.DataFrame(deviation_data)

    st.markdown("## Best model summary")

    # Get best model
    best_model_row = df_deviation.loc[df_deviation["Average deviation"].idxmin()]
    best_model_name = best_model_row["Model name"]

    st.markdown(f"The best model is **{best_model_name}** based on average deviation.")
