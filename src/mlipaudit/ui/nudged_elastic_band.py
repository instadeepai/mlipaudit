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

from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    NEBResult,
    NudgedElasticBandBenchmark,
)
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, NEBResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data = []
    model_names = []
    for model_name, results in data.items():
        if model_name in selected_models:
            model_data_converted = {
                "Score": results.score,
                "Convergence rate": results.convergence_rate,
            }
            converted_data.append(model_data_converted)
            model_names.append(model_name)

    return pd.DataFrame(converted_data, index=model_names)


def nudged_elastic_band_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the nudged elastic band convergence benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Nudged Elastic Band")

    st.markdown(
        "The nudged elastic band (NEB) is a method to relax a mean energy path between "
        "a reactant and a product structure and thereby find a good guess for the "
        "transition state. Here, the benchmark only assesses if the model is able to "
        "converge the NEB calculations with a known transition state."
    )

    st.markdown(
        "Reactant and product structures are energy minimized using the MLIP. Then, an "
        "initial guess for the mean energy path is constructed placing the known "
        "transition state structure in the middle. The path is then relaxed using "
        "two NEB runs, the second using the climbing image."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit/"
        "benchmarks/small_molecules/nudged_elastic_band.html)."
    )

    # Download data and get model names
    if "nudged_elastic_band_cached_data" not in st.session_state:
        st.session_state.nudged_elastic_band_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = (
        st.session_state.nudged_elastic_band_cached_data
    )

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

    df = _process_data_into_dataframe(data, selected_models)
    df_display = df.copy()
    df_display.index.name = "Model name"

    df_display.sort_values("Score", ascending=False, inplace=True)

    st.markdown("## Summary statistics")

    display_model_scores(df_display)

    st.markdown("## Convergence percentage")
    st.markdown("")

    chart_df = (
        df.reset_index()
        .melt(
            id_vars=["index"],
            value_vars=["Convergence rate"],
            var_name="Metric",
            value_name="Value",
        )
        .rename(columns={"index": "Model"})
    )

    chart_df = chart_df.sort_values("Value", ascending=False)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model:N", title="Model", axis=alt.Axis(labelAngle=-45, labelLimit=100)
            ),
            y=alt.Y("Value:Q", title="Convergence (%)"),
            color=alt.Color("Model:N", title="Model"),
        )
        .properties(width=600, height=400)
    )
    st.altair_chart(chart, use_container_width=True)


class ConformerSelectionPageWrapper(UIPageWrapper):
    """Page wrapper for conformer selection benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return nudged_elastic_band_page

    @classmethod
    def get_benchmark_class(cls) -> type[NudgedElasticBandBenchmark]:  # noqa: D102
        return NudgedElasticBandBenchmark
