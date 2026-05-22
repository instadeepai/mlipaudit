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

from mlipaudit.benchmarks import ScalingBenchmark, ScalingResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, ScalingResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels, selected_models: list[str]
) -> pd.DataFrame:
    df_data = []
    for model_name, result in data.items():
        if model_name in selected_models:
            for structure_result in result.structures:
                if structure_result.failed:
                    continue

                df_data.append({
                    "Model name": model_name,
                    "Structure": structure_result.structure_name,
                    "Average episode time (s)": structure_result.average_episode_time,
                    "Num atoms": structure_result.num_atoms,
                    "Num steps": structure_result.num_steps,
                    "Num episodes": structure_result.num_episodes,
                    "Average step time (s)": structure_result.average_step_time,
                })
    return pd.DataFrame(df_data)


def plot_all_models_performance(df: pd.DataFrame) -> alt.Chart:
    """Plot the scaling curves for all models together.

    Args:
        df: The dataframe containing the inference times for all models.

    Returns:
        The Altair chart.
    """
    # Create base chart
    base = alt.Chart(df).encode(
        x=alt.X("Num atoms:Q", title="System size (number of atoms)"),
        y=alt.Y("Average step time (s):Q", title="Average step time (s)"),
        color=alt.Color(
            "Model name:N", title="Model", legend=alt.Legend(title="Model")
        ),
        tooltip=[
            alt.Tooltip("Model name:N", title="Model"),
            alt.Tooltip("Structure:N", title="Structure"),
            alt.Tooltip("Num atoms:Q", title="Number of atoms"),
            alt.Tooltip(
                "Average step time (s):Q", title="Average step time (s)", format=".4f"
            ),
        ],
    )

    # Create scatter plot
    scatter = base.mark_point(size=60, opacity=0.7)

    # Create regression lines for each model
    regression_lines = base.transform_regression(
        on="Num atoms",
        regression="Average step time (s)",
        method="linear",
        groupby=["Model name"],
    ).mark_line(strokeWidth=2)

    # Combine scatter and regression lines
    chart = (scatter + regression_lines).properties(
        width=800,
        height=500,
    )

    st.altair_chart(chart, use_container_width=True)
    return chart


def scaling_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the scaling page.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Inference scaling")

    st.markdown(
        "This module assesses the scaling of MLIPs with respect "
        "to molecule size. Simulations are run for a single episode for a "
        "relatively large number of steps on several systems of varying sizes."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/general/scaling.html)."
    )

    # Download data and get model names
    if "scaling_data" not in st.session_state:
        st.session_state.scaling_data = data_func()

    # Retrieve the data from the session state
    data = st.session_state.scaling_data

    if not data:
        st.markdown("**No results to display**.")
        return

    failed_models = get_failed_models(data)
    display_failed_models(failed_models)
    data = filter_failed_results(data)

    st.markdown("## Inference scaling: Average step time vs system size")

    selected_models = fetch_selected_models(available_models=list(data.keys()))

    if not selected_models:
        st.markdown("**No results to display**.")
        return

    df = _process_data_into_dataframe(data, selected_models)

    chart = plot_all_models_performance(df)  # noqa: F841


class ScalingPageWrapper(UIPageWrapper):
    """Page wrapper for scaling benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return scaling_page

    @classmethod
    def get_benchmark_class(cls) -> type[ScalingBenchmark]:  # noqa: D102
        return ScalingBenchmark
