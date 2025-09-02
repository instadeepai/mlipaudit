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


import io
import statistics
from typing import Callable, TypeAlias

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.ring_planarity.ring_planarity import RingPlanarityResult

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
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit-open/benchmarks/small-molecules/ring_planarity.html)."
    )

    # Download data and get model names
    if "ring_planarity_cached_data" not in st.session_state:
        st.session_state.ring_planarity_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.ring_planarity_cached_data

    unique_model_names = list(set(data.keys()))
    model_select = st.sidebar.multiselect(
        "Select model(s)", unique_model_names, default=unique_model_names
    )
    selected_models = model_select if model_select else unique_model_names

    deviation_data = [
        {
            "Model name": model_name,
            "Average deviations": [
                mol_result.avg_deviation for mol_result in result.molecules
            ],
            "Average deviation": statistics.mean(
                mol_result.avg_deviation for mol_result in result.molecules
            ),
        }
        for model_name, result in data.items()
        if model_name in selected_models
    ]

    df_deviation = pd.DataFrame(deviation_data)

    st.markdown("## Best model summary")

    # Get best model
    best_model_row = df_deviation.loc[df_deviation["Average deviation"].idxmin()]
    best_model_name = best_model_row["Model name"]

    st.markdown(f"The best model is **{best_model_name}** based on average deviation.")

    st.metric(
        "Total average deviation",
        f"{float(best_model_row['Average deviation']):.3f}",
    )

    st.markdown("## Ring planarity deviation distribution per model")

    # Get all unique ring types from the data
    all_ring_types_set: set[str] = set()

    for model_name, result in data.items():
        all_ring_types_set.update(mol.molecule_name for mol in result.molecules)
    all_ring_types = sorted(list(all_ring_types_set))

    # Ring type selection dropdown
    selected_ring_type = st.selectbox(
        "Select ring type:", all_ring_types, index=0 if all_ring_types else None
    )

    if selected_ring_type:
        # Transform the data for the selected ring type
        plot_data = []

        for model_name, result in data.items():
            for mol in result.molecules:
                if selected_ring_type == mol.molecule_name:
                    for ring_deviation in mol.deviation_trajectory:
                        plot_data.append({
                            "Model name": model_name,
                            "Ring deviation": ring_deviation,
                        })

        df_plot = pd.DataFrame(plot_data)

        # Create the boxplot chart
        chart = (
            alt.Chart(df_plot)
            .mark_boxplot(extent="min-max", size=50, color="darkgrey")
            .encode(
                x=alt.X(
                    "Model name:N",
                    title="Model name",
                    axis=alt.Axis(labelAngle=-45, labelLimit=100),
                ),
                y=alt.Y(
                    "Ring deviation:Q",
                    title="Ring deviation (Å)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Model name:N",
                    title="Model name",
                    legend=alt.Legend(orient="top"),
                ),
            )
            .properties(
                width=600,
                height=400,
            )
        )

        st.altair_chart(chart, use_container_width=True)
        buffer = io.BytesIO()
        chart.save(buffer, format="png", ppi=300)
        img_bytes = buffer.getvalue()
        st.download_button(
            label="Download plot",
            data=img_bytes,
            file_name="small_molecule_ring_planarity_chart.png",
        )
    else:
        st.info("Please select a ring type to view the distribution.")
