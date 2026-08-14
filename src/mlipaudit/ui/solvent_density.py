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

from mlipaudit.benchmarks import SolventDensityBenchmark, SolventDensityResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
    ordered_structure_names,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, SolventDensityResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data_scores = []
    for model_name, result in data.items():
        if model_name in selected_models:
            model_data_converted = {
                "Model name": model_name,
                "Score": result.score,
                "Average density deviation (g/cm3)": result.avg_density_deviation,
            }
            for structure_res in result.structures:
                if structure_res.failed:
                    continue

                model_data_converted[
                    f"{structure_res.structure_name} density (g/cm3)"
                ] = structure_res.average_density

                model_data_converted[
                    f"{structure_res.structure_name} density deviation (g/cm3)"
                ] = structure_res.density_deviation
            converted_data_scores.append(model_data_converted)
    df = pd.DataFrame(converted_data_scores)
    return df


def solvent_density_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the solvent density benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Solvent density")

    st.markdown(
        "Here we show the equilibrium density of each molecular solvent, obtained from "
        "NPT simulations. The dashed lines show the reference density of each solvent. "
        "A box that expands or collapses will show up as a large deviation from the "
        "reference density."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit/"
        "benchmarks/molecular_liquids/density.html)."
    )

    if "solvent_density_cached_data" not in st.session_state:
        st.session_state.solvent_density_cached_data = data_func()

    data: BenchmarkResultForMultipleModels = (
        st.session_state.solvent_density_cached_data
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

    st.markdown("## Summary statistics")

    df = _process_data_into_dataframe(data, selected_models)
    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## Density time series")

    for solvent in ordered_structure_names(data, selected_models):
        plot_data_solvent = []
        reference_density = None

        for model_name, result in data.items():
            if model_name not in selected_models:
                continue
            structure_res = {s.structure_name: s for s in result.structures}.get(
                solvent
            )
            if (
                structure_res is None
                or structure_res.failed
                or structure_res.densities is None
            ):
                continue

            reference_density = structure_res.reference_density
            for frame, density in enumerate(structure_res.densities):
                plot_data_solvent.append({
                    "Frame": frame,
                    "Density (g/cm3)": density,
                    "model": str(model_name),
                })

        if not plot_data_solvent:
            st.warning(f"No data found for {solvent}")
            continue

        st.subheader(f"Density of {solvent}")

        df_plot_solvent = pd.DataFrame(plot_data_solvent)

        chart_solvent = (
            alt.Chart(df_plot_solvent)
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
            chart_solvent = chart_solvent + reference_line

        st.altair_chart(chart_solvent, width="stretch")


class SolventDensityPageWrapper(UIPageWrapper):
    """Page wrapper for the solvent density benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return solvent_density_page

    @classmethod
    def get_benchmark_class(cls) -> type[SolventDensityBenchmark]:  # noqa: D102
        return SolventDensityBenchmark
