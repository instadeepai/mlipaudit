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

import json
from pathlib import Path
from typing import Callable, TypeAlias

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.npyio import NpzFile

from mlipaudit.benchmarks import (
    SolventRadialDistributionBenchmark,
    SolventRadialDistributionResult,
)
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

APP_DATA_DIR = Path(__file__).parent.parent / "app_data"
SOLVENT_RADIAL_DISTRIBUTION_DATA_DIR = APP_DATA_DIR / "solvent_radial_distribution"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, SolventRadialDistributionResult
]

RADIUS_CUTOFF = 12


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
                "Average peak deviation (Å)": result.avg_peak_deviation,
            }
            for structure_res in result.structures:
                if structure_res.failed:
                    continue

                model_data_converted[
                    f"{structure_res.structure_name} peak deviation (Å)"
                ] = structure_res.peak_deviation
            converted_data_scores.append(model_data_converted)
    df = pd.DataFrame(converted_data_scores)
    return df


@st.cache_resource
def _load_experimental_data() -> NpzFile:
    with open(
        SOLVENT_RADIAL_DISTRIBUTION_DATA_DIR / "solvent_maxima_experimental.json",
        "r",
        encoding="utf-8",
    ) as f:
        solvent_maxima = json.load(f)
        return solvent_maxima


def solvent_radial_distribution_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the solvent rdf page.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Solvent Radial distribution function")

    st.markdown(
        "Here we show the radial distribution function of the solvents CCl4, "
        "methanol, and acetonitrile. The vertical lines show the reference "
        "maximum of the radial distribution function for each solvent."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit/"
        "benchmarks/molecular_liquids/radial_distribution.html)."
    )

    # Download data and get model names
    if "solvent_radial_distribution_cached_data" not in st.session_state:
        st.session_state.solvent_radial_distribution_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = (
        st.session_state.solvent_radial_distribution_cached_data
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

    solvent_maxima = _load_experimental_data()

    st.markdown("## Summary statistics")

    df = _process_data_into_dataframe(data, selected_models)

    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## Radial distribution functions")

    for solvent_index, solvent in enumerate(["CCl4", "methanol", "acetonitrile"]):
        rdf_data_solvent = {}

        for model_name, result in data.items():
            if (
                model_name in selected_models
                and solvent in result.structure_names
                and not result.structures[solvent_index].failed
            ):
                rdf_data_solvent[model_name] = {
                    "r": np.array(result.structures[solvent_index].radii),
                    "rdf": np.array(result.structures[solvent_index].rdf),
                }

        if len(rdf_data_solvent) > 0:
            st.subheader(
                f"Radial distribution function of {solvent} "
                f"({solvent_maxima[solvent]['type']})"
            )

            # Convert to long format for Altair plotting
            plot_data_solvent = []
            for model_name, model_data in rdf_data_solvent.items():
                r_values = model_data["r"]
                rdf_values = model_data["rdf"]
                for r_val, rdf_val in zip(r_values, rdf_values):
                    # Only include data points where r < 12
                    if r_val < RADIUS_CUTOFF:
                        plot_data_solvent.append({
                            "r": r_val,
                            "rdf": rdf_val,
                            "model": str(model_name),
                        })

            df_plot_solvent = pd.DataFrame(plot_data_solvent)

            # Create Altair line chart for solvent
            chart_solvent = (
                alt.Chart(df_plot_solvent)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("r:Q", title="Distance r (Å)"),
                    y=alt.Y("rdf:Q", title="Radial Distribution Function"),
                    color=alt.Color("model:N", title="Model"),
                )
                .properties(width=800, height=400)
            )

            # Add vertical line at experimental maximum
            vline = (
                alt.Chart(pd.DataFrame({"x": [solvent_maxima[solvent]["distance"]]}))
                .mark_rule(color="black", strokeWidth=2)
                .encode(x="x:Q")
            )

            # Combine the line chart with the vertical line
            combined_chart = chart_solvent + vline

            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.warning(f"No data found for {solvent}")


class SolventRadialDistributionPageWrapper(UIPageWrapper):
    """Page wrapper for solvent radial distribution benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return solvent_radial_distribution_page

    @classmethod
    def get_benchmark_class(cls) -> type[SolventRadialDistributionBenchmark]:  # noqa: D102
        return SolventRadialDistributionBenchmark
