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
from ase import units

from mlipaudit.benchmarks import TautomersBenchmark, TautomersResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, TautomersResult]


def tautomers_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the tautomers benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Tautomers")

    st.markdown(
        "Tautomers are isomers that can interconvert by the movement "
        "of a proton and/or the rearrangement of double bonds. "
        "This benchmark evaluates "
        "how well MLIPs can predict the relative energies and stability of different "
        "tautomeric forms of molecules in-vacuum. "
        "The dataset contains 1391 tautomer pairs with reference QM energies extracted "
        "from the [Tautobase](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00035) "
        "dataset."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io"
        "/mlipaudit/benchmarks/small_molecules/tautomers.html)."
    )

    with st.sidebar.container():
        selected_energy_unit = st.selectbox(
            "Select an energy unit:",
            ["kcal/mol", "eV"],
        )

    conversion_factor = (
        1.0 if selected_energy_unit == "kcal/mol" else (units.kcal / units.mol)
    )

    # Download data and get model names
    if "tautomers_cached_data" not in st.session_state:
        st.session_state.tautomers_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.tautomers_cached_data

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

    # Calculate MAE and RMSE for each model_id
    metrics_data = []
    for model_name in selected_models:
        mae = data[model_name].mae * conversion_factor  # type: ignore
        rmse = data[model_name].rmse * conversion_factor  # type: ignore
        metrics_data.extend([
            {"Model name": model_name, "metric": "MAE", "value": mae},
            {"Model name": model_name, "metric": "RMSE", "value": rmse},
        ])

    df_summary = pd.DataFrame([
        {
            "Model name": model_name,
            "Score": result.score,
            f"MAE ({selected_energy_unit})": result.mae,
            f"RMSE ({selected_energy_unit})": result.rmse,
        }
        for model_name, result in data.items()
        if model_name in selected_models
    ])
    st.markdown("## Summary statistics")

    df_summary.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df_summary)

    metrics_df = pd.DataFrame(metrics_data)

    # Create grouped bar chart
    st.markdown("## MAE and RMSE by model")
    chart = (
        alt.Chart(metrics_df)
        .mark_bar()
        .add_selection(alt.selection_interval())
        .encode(
            x=alt.X(
                "Model name:N",
                title="Model",
                axis=alt.Axis(labelAngle=-45, labelLimit=100),
            ),
            y=alt.Y("value:Q", title=f"Error ({selected_energy_unit})"),
            color=alt.Color(
                "metric:N",
                title="Metric",
            ),
            xOffset=alt.XOffset("metric:N"),
            tooltip=[
                alt.Tooltip("Model name:N", title="Model"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Error"),
            ],
        )
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    @st.cache_data
    def convert_for_download(df):
        return df.to_csv().encode("utf-8")

    # Convert to long-format DataFrame
    converted_data = []
    for model_name, result in data.items():
        for molecule in result.molecules:
            if molecule.failed:
                continue

            converted_data.append({
                "Model name": model_name,
                "Score": result.score,
                "structure ID": molecule.structure_id,
                "abs_deviation": molecule.abs_deviation * conversion_factor  # type: ignore
                if not molecule.failed
                else None,
                "pred_energy_diff": molecule.predicted_energy_diff * conversion_factor  # type: ignore
                if not molecule.failed
                else None,
                "ref_energy_diff": molecule.ref_energy_diff * conversion_factor  # type: ignore
                if not molecule.failed
                else None,
            })

    df_detailed = pd.DataFrame(converted_data)

    csv = convert_for_download(df_detailed)
    st.download_button(
        label="Download full table as CSV",
        data=csv,
        file_name="tautomers_data.csv",
    )


class TautomersPageWrapper(UIPageWrapper):
    """Page wrapper for tautomers benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return tautomers_page

    @classmethod
    def get_benchmark_class(cls) -> type[TautomersBenchmark]:  # noqa: D102
        return TautomersBenchmark
