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

from mlipaudit.reactivity import (
    ReactivityResult,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, ReactivityResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
    conversion_factor: float,
) -> pd.DataFrame:
    converted_data_scores, model_names = [], []
    for model_name, result in data.items():
        if model_name in selected_models:
            model_data_converted = {
                "Activation energy MAE": result.mae_activation_energy
                * conversion_factor,
                "Activation energy RMSE": result.rmse_activation_energy
                * conversion_factor,
                "Enthalpy of reaction MAE": result.mae_enthalpy_of_reaction
                * conversion_factor,
                "Enthalpy of reaction RMSE": result.mae_enthalpy_of_reaction
                * conversion_factor,
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
    st.sidebar.markdown("# Reactivity")

    st.markdown(
        "This benchmarks assesses the **MLIP**'s capability to predict"
        " the energy of transition states and thereby the activation"
        " energy and enthalpy of formation of a reaction. Accurately"
        " modeling chemical reactions is an important use case to employ"
        " MLIPs to understand reactivity and to predict the outcomes of"
        " chemical reactions."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit-open/benchmarks/small_molecules/reactivity.html)."
    )

    # Download data and get model names
    if "reactivity_cached_data" not in st.session_state:
        st.session_state.reactivity_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.reactivity_cached_data

    unique_model_names = list(set(data.keys()))
    model_select = st.sidebar.multiselect(
        "Select model(s)", unique_model_names, default=unique_model_names
    )
    selected_models = model_select if model_select else unique_model_names

    with st.sidebar.container():
        unit_selection = st.selectbox(
            "Select an energy unit:",
            ["kcal/mol", "eV"],
        )

    # Set conversion factor based on selection
    if unit_selection == "kcal/mol":
        conversion_factor = 1.0
    else:
        conversion_factor = units.kcal / units.mol

    df = _process_data_into_dataframe(data, selected_models, conversion_factor)

    st.markdown("## Best model summary")

    # Get best model
    best_model_row = df.loc[df["Activation energy MAE"].idxmin()]
    best_model_name = best_model_row.name

    st.markdown(
        f"The best model is **{best_model_name}** based on "
        "the RMSE of the activation energy."
    )

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Activation Energy MAE",
            value=f"{best_model_row['Activation energy MAE']:.3f}",
        )

    with col2:
        st.metric(
            label="Activation Energy RMSE",
            value=f"{best_model_row['Activation energy RMSE']:.3f}",
        )

    with col3:
        st.metric(
            label="Enthalpy of Reaction MAE",
            value=f"{best_model_row['Enthalpy of reaction MAE']:.3f}",
        )

    with col4:
        st.metric(
            label="Enthalpy of Reaction RMSE",
            value=f"{best_model_row['Enthalpy of reaction RMSE']:.3f}",
        )

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
                f"Activation energy {selected_metric}",
                f"Enthalpy of reaction {selected_metric}",
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
            y=alt.Y("value:Q", title=f"{selected_metric} ({unit_selection})"),
            color=alt.Color("Model name:N", title="Model name"),
            xOffset="Model name:N",
        )
        .properties(width=600, height=400)
        .resolve_scale(color="independent")
    )

    st.altair_chart(chart, use_container_width=True)
