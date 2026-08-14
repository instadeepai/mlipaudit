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

from mlipaudit.benchmarks import WaterDensityBenchmark, WaterDensityResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, WaterDensityResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data_scores, model_names = [], []
    for model_name, result in data.items():
        if model_name in selected_models:
            converted_data_scores.append({
                "Score": result.score,
                "Equilibrium density (g/cm3)": result.average_density,
                "Density deviation (g/cm3)": result.density_deviation,
                "Reference density (g/cm3)": result.reference_density,
            })
            model_names.append(model_name)

    df = pd.DataFrame(converted_data_scores, index=model_names)
    return df


def water_density_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the water density benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Water density")

    st.markdown(
        "The equilibrium density of liquid water is a fundamental property that a "
        "good MLIP should reproduce. We run an NPT simulation of a box of water "
        "molecules and compare the resulting equilibrium density to the experimental "
        "reference. A box that expands or collapses will show up as a large deviation "
        "from the reference density."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/molecular_liquids/density.html)."
    )

    if "water_density_cached_data" not in st.session_state:
        st.session_state.water_density_cached_data = data_func()

    data: BenchmarkResultForMultipleModels = st.session_state.water_density_cached_data

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

    st.markdown("## Summary statistics")

    df = _process_data_into_dataframe(data, selected_models)
    df = df.rename_axis("Model name")
    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## Density time series")
    st.markdown(
        "Here we show the density of the water box over the course of the simulation. "
        "The black dashed line shows the experimental reference density. A "
        "well-behaved simulation should equilibrate close to this line."
    )

    plot_data = []
    reference_density = None
    for model_name, result in data.items():
        if (
            model_name in selected_models
            and not result.failed
            and result.densities is not None
        ):
            reference_density = result.reference_density
            for frame, density in enumerate(result.densities):
                plot_data.append({
                    "Frame": frame,
                    "Density (g/cm3)": density,
                    "model": str(model_name),
                })

    if not plot_data:
        st.markdown("**No density time series to display**.")
        return

    df_plot = pd.DataFrame(plot_data)

    chart = (
        alt.Chart(df_plot)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("Frame:Q", title="Frame"),
            y=alt.Y(
                "Density (g/cm3):Q",
                title="Density (g/cm³)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("model:N", title="Model"),
        )
        .properties(width=800, height=400)
    )

    if reference_density is not None:
        reference_line = (
            alt.Chart(pd.DataFrame({"y": [reference_density]}))
            .mark_rule(color="black", strokeDash=[6, 4], strokeWidth=2)
            .encode(y="y:Q")
        )
        chart = chart + reference_line

    st.altair_chart(chart, width="stretch")


class WaterDensityPageWrapper(UIPageWrapper):
    """Page wrapper for the water density benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return water_density_page

    @classmethod
    def get_benchmark_class(cls) -> type[WaterDensityBenchmark]:  # noqa: D102
        return WaterDensityBenchmark
