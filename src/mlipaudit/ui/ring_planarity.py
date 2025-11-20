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

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import RingPlanarityBenchmark, RingPlanarityResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

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
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/small_molecules/ring_planarity.html)."
    )

    # Download data and get model names
    if "ring_planarity_cached_data" not in st.session_state:
        st.session_state.ring_planarity_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.ring_planarity_cached_data

    if not data:
        st.markdown("**No results to display**.")
        return

    selected_models = fetch_selected_models(available_models=list(data.keys()))

    if not selected_models:
        st.markdown("**No results to display**.")
        return

    failed_models = get_failed_models(data)
    display_failed_models(failed_models)
    data = filter_failed_results(data)

    deviation_data = [
        {
            "Model name": model_name,
            "Score": result.score,
            "Average deviation (Å)": result.mae_deviation,
        }
        for model_name, result in data.items()
        if model_name in selected_models
    ]

    df_deviation = pd.DataFrame(deviation_data)

    st.markdown("## Summary statistics")

    df_deviation.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df_deviation)

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
            if model_name in selected_models:
                for mol in result.molecules:
                    if selected_ring_type == mol.molecule_name and not mol.failed:
                        for ring_deviation in mol.deviation_trajectory:  # type: ignore
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
                    title="Model",
                    axis=alt.Axis(labelAngle=-45, labelLimit=100),
                ),
                y=alt.Y(
                    "Ring deviation:Q",
                    title="Ring deviation (Å)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Model name:N",
                    title="Model",
                    legend=alt.Legend(orient="top"),
                ),
            )
            .properties(
                width=600,
                height=400,
            )
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Please select a ring type to view the distribution.")


class RingPlanarityPageWrapper(UIPageWrapper):
    """Page wrapper for ring planarity benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return ring_planarity_page

    @classmethod
    def get_benchmark_class(cls) -> type[RingPlanarityBenchmark]:  # noqa: D102
        return RingPlanarityBenchmark
