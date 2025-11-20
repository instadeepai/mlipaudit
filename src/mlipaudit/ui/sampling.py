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

from collections import defaultdict
from typing import Callable, TypeAlias

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import SamplingBenchmark, SamplingResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, SamplingResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data_scores = []
    model_names = []
    for model_name, result in data.items():
        if model_name in selected_models:
            model_data_converted = {
                "Score": result.score,
                "Backbone Distribution RMSD (Å)": result.rmsd_backbone_total,
                "Backbone Distribution Hellinger Distance": (
                    result.hellinger_distance_backbone_total
                ),
                "Sidechain Distribution RMSD (Å)": result.rmsd_sidechain_total,
                "Sidechain Distribution Hellinger Distance": (
                    result.hellinger_distance_sidechain_total
                ),
                "Outliers Ratio Backbone": result.outliers_ratio_backbone_total,
                "Outliers Ratio Sidechain": result.outliers_ratio_sidechain_total,
                "Number of Exploded Systems": len(result.exploded_systems),
            }

            converted_data_scores.append(model_data_converted)
            model_names.append(model_name)

    return pd.DataFrame(converted_data_scores, index=model_names)


def _process_data_into_dataframe_per_residue(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
    metric_option: str,
) -> pd.DataFrame:
    converted_data_scores = []
    model_index = []
    for model_name, results in data.items():
        if model_name in selected_models:
            model_data_converted = defaultdict(float)
            residue_types = list(results.rmsd_backbone_dihedrals.keys())  # type: ignore
            for residue_type in residue_types:
                rmsd = results.rmsd_backbone_dihedrals[residue_type]  # type: ignore
                hellinger = results.hellinger_distance_backbone_dihedrals[  # type: ignore
                    residue_type
                ]
                if metric_option == "RMSD":
                    model_data_converted[residue_type] = rmsd

                else:
                    model_data_converted[residue_type] = hellinger

            converted_data_scores.append(model_data_converted)
            model_index.append(model_name)

    return pd.DataFrame(converted_data_scores, index=model_index).T


def sampling_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the sampling benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Protein Conformational Sampling")

    st.markdown(
        (
            "This benchmark evaluates MLIP quality by analyzing protein sampling in MD "
            "simulations. It computes backbone Ramachandran and side-chain "
            "rotamer angles, and identifies structural outliers by comparison to "
            "reference rotamer libraries."
        )
    )

    st.markdown(
        (
            "For more information, see the "
            "[docs](https://instadeepai.github.io/mlipaudit"
            "/benchmarks/biomolecules/sampling.html)."
        )
    )

    if "sampling_cached_data" not in st.session_state:
        st.session_state.sampling_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.sampling_cached_data

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
    df_summary = df.copy()
    df_summary = df_summary.rename_axis("Model name")
    df_summary.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df_summary)

    df_noexploded = df[df["Number of Exploded Systems"] == 0]
    if len(df_noexploded) == 0:
        st.markdown(
            "None of the models were able to sample all systems without exploding."
        )

    st.markdown("## Backbone and sidechain distribution metrics")

    metric_option = st.selectbox(
        "Select metric to display",
        options=["RMSD", "Hellinger distance"],
        index=0,
        key="sampling_metric_select",
    )

    if metric_option == "RMSD":
        values = ["Backbone Distribution RMSD (Å)", "Sidechain Distribution RMSD (Å)"]
    else:
        values = [
            "Backbone Distribution Hellinger Distance",
            "Sidechain Distribution Hellinger Distance",
        ]

    chart_df = (
        df.reset_index()
        .melt(
            id_vars=["index"],
            value_vars=values,
            var_name="Metric",
            value_name="Value",
        )
        .rename(columns={"index": "Model"})
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model:N", title="Model", axis=alt.Axis(labelAngle=-45, labelLimit=100)
            ),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Metric:N", title="Metric"),
            xOffset="Metric:N",
        )
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("## Per-residue distribution metrics")

    metric_option = st.selectbox(
        "Select metric to display",
        options=["RMSD", "Hellinger distance"],
        index=0,
        key="sampling_metric_select_per_residue",
    )

    df_per_residue = _process_data_into_dataframe_per_residue(
        data, selected_models, metric_option
    )
    df_per_residue_display = df_per_residue.style.background_gradient(
        vmin=df_per_residue.min().min(),
        vmax=df_per_residue.max().max(),
    ).format(precision=3)
    st.dataframe(df_per_residue_display)

    st.markdown("## Outliers")
    st.markdown(
        "Here we show how many of the sampled dihedrals are outliers, meaning that "
        "they are far away from any point of the reference data."
    )

    chart_df_outliers = (
        df.reset_index()
        .melt(
            id_vars=["index"],
            value_vars=["Outliers Ratio Backbone", "Outliers Ratio Sidechain"],
            var_name="Metric",
            value_name="Value",
        )
        .rename(columns={"index": "Model"})
    )

    if sum(chart_df_outliers["Value"]) == 0:
        st.markdown(
            "**No outliers found:** "
            "All sampled dihedrals are close to the reference data."
        )
    else:
        chart_outliers = (
            alt.Chart(chart_df_outliers)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Model:N",
                    title="Model",
                    axis=alt.Axis(labelAngle=-45, labelLimit=100),
                ),
                y=alt.Y("Value:Q", title="Outliers ratio"),
                color=alt.Color("Metric:N", title="Metric"),
                xOffset="Metric:N",
            )
            .properties(width=600, height=400)
        )

        st.altair_chart(chart_outliers, use_container_width=True)


class SamplingPageWrapper(UIPageWrapper):
    """Page wrapper for sampling benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return sampling_page

    @classmethod
    def get_benchmark_class(cls) -> type[SamplingBenchmark]:  # noqa: D102
        return SamplingBenchmark
