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

import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import StabilityBenchmark, StabilityResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, StabilityResult]

FS_TO_NS = 1e-6


def _process_data_into_dataframe(
    data: dict[str, StabilityResult], selected_models: list[str]
) -> pd.DataFrame:
    df_data = []
    for model_name, result in data.items():
        if model_name in selected_models:
            for structure_result in result.structure_results:
                if structure_result.failed:
                    continue

                sim_duration_ns = (
                    structure_result.num_steps * FS_TO_NS
                )  # Convert from fs to ns
                df_data.append({
                    "Model name": model_name,
                    "Structure": structure_result.structure_name,
                    "Stable": True
                    if structure_result.exploded_frame == -1
                    and structure_result.drift_frame == -1
                    else False,
                    "Score": structure_result.score,
                    "Explosion time": (
                        structure_result.exploded_frame / structure_result.num_frames
                    )
                    * sim_duration_ns
                    if structure_result.exploded_frame != -1
                    else None,
                    "Hydrogen drit time": (
                        structure_result.drift_frame / structure_result.num_frames
                    )
                    * sim_duration_ns
                    if structure_result.drift_frame != -1
                    else None,
                    "Simulation duration (ns)": f"{sim_duration_ns:.3f}",
                })

    return pd.DataFrame(df_data)


def stability_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the stability benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Stability")

    st.markdown(
        "This module assesses the stability of MLIPs by running molecular "
        "dynamics simulations on large, biologically relevant assemblies and checking "
        "for stability metrics."
    )

    st.markdown(
        "For more information, see the [docs]"
        "(https:/instadeepai.github.io/mlipaudit/benchmarks/general/stability.html)."
    )

    # Download data and get model names
    if "stability_cached_data" not in st.session_state:
        st.session_state.stability_cached_data = data_func()

    # Retrieve the data from the session state
    data = st.session_state.stability_cached_data

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

    df_avg_score = pd.DataFrame(
        {"Model name": model_name, "Score": result.score}
        for model_name, result in data.items()
        if model_name in selected_models
    )

    # Apply conditional styling for NA values
    def style_na_values(val):
        """Style function to color values green.

        Returns:
            str: green background if value is None.
        """
        if pd.isnull(val):
            return (
                "background-color: #90EE90;"  # Light green background, dark green text
            )
        return ""

    # Apply styling to specific columns
    df.style.map(style_na_values, subset=["Explosion time", "Hydrogen drift time"])

    # Find models that are stable for ALL structures
    stable_models = []
    for model_name in df["Model name"].unique():
        model_data = df[df["Model name"] == model_name]
        if all(model_data["Stable"]):
            stable_models.append(model_name)

    st.markdown("## Summary statistics")

    df_avg_score.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df_avg_score)

    st.markdown("## Stability per model and structure")
    # Display the styled DataFrame with column configuration
    st.dataframe(
        df.style.format(precision=3),
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=1,
                format="%.2f",
            )
        },
    )


class StabilityPageWrapper(UIPageWrapper):
    """Page wrapper for stability benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return stability_page

    @classmethod
    def get_benchmark_class(cls) -> type[StabilityBenchmark]:  # noqa: D102
        return StabilityBenchmark
