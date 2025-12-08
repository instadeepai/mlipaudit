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

from mlipaudit.benchmarks import ReactivityBenchmark, ReactivityResult
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, ReactivityResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
    conversion_factor: float,
    selected_energy_unit: str,
) -> pd.DataFrame:
    converted_data_scores, model_names = [], []
    for model_name, result in data.items():
        if model_name in selected_models:
            mae_activation = result.mae_activation_energy * conversion_factor  # type: ignore
            rmse_activation = result.rmse_activation_energy * conversion_factor  # type: ignore
            mae_enthalpy = result.mae_enthalpy_of_reaction * conversion_factor  # type: ignore
            rmse_enthalpy = result.rmse_enthalpy_of_reaction * conversion_factor  # type: ignore
            model_data_converted = {
                "Score": result.score,
                f"Activation energy MAE ({selected_energy_unit})": mae_activation,
                f"Activation energy RMSE ({selected_energy_unit})": rmse_activation,
                f"Enthalpy of reaction MAE ({selected_energy_unit})": mae_enthalpy,
                f"Enthalpy of reaction RMSE ({selected_energy_unit})": rmse_enthalpy,
            }
            converted_data_scores.append(model_data_converted)
            model_names.append(model_name)

    return pd.DataFrame(converted_data_scores, index=model_names)


def reactivity_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for reactivity.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Reactivity")

    st.markdown(
        "This benchmarks assesses the MLIP's capability to predict"
        " the energy of transition states and thereby the activation"
        " energy and enthalpy of formation of a reaction. Accurately"
        " modeling chemical reactions is an important use case to employ"
        " MLIPs to understand reactivity and to predict the outcomes of"
        " chemical reactions."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/small_molecules/reactivity.html)."
    )

    # Download data and get model names
    if "reactivity_cached_data" not in st.session_state:
        st.session_state.reactivity_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.reactivity_cached_data

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

    with st.sidebar.container():
        selected_energy_unit = st.selectbox(
            "Select an energy unit:",
            ["kcal/mol", "eV"],
        )

    conversion_factor = (
        1.0 if selected_energy_unit == "kcal/mol" else (units.kcal / units.mol)
    )

    df = _process_data_into_dataframe(
        data, selected_models, conversion_factor, selected_energy_unit
    )

    st.markdown("## Summary statistics")

    df.sort_values("Score", ascending=False, inplace=True)
    df = df.rename_axis("Model name")
    display_model_scores(df)

    st.markdown("## Activation energy and enthalpy of reaction errors")

    # Create dropdown for error metric selection
    selected_metric = st.selectbox(
        "Select error metric:", ["MAE", "RMSE"], key="metric_selector"
    )

    df_melted = (
        df.melt(
            ignore_index=False,
            var_name="Metric Type",
            value_vars=[
                f"Activation energy {selected_metric} ({selected_energy_unit})",
                f"Enthalpy of reaction {selected_metric} ({selected_energy_unit})",
            ],
        )
        .reset_index()
        .rename(columns={"index": "Model name"})
    )

    # Create the bar chart
    chart = (
        alt.Chart(df_melted)
        .mark_bar()
        .encode(
            x=alt.X("Metric Type:N", title="Energy Type", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=f"{selected_metric} ({selected_energy_unit})"),
            color=alt.Color("Model name:N", title="Model"),
            xOffset="Model name:N",
        )
        .properties(width=600, height=400)
        .resolve_scale(color="independent")
    )

    st.altair_chart(chart, use_container_width=True)


class ReactivityPageWrapper(UIPageWrapper):
    """Page wrapper for reactivity page."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return reactivity_page

    @classmethod
    def get_benchmark_class(cls) -> type[ReactivityBenchmark]:  # noqa: D102
        return ReactivityBenchmark
