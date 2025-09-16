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

from pathlib import Path
from typing import Callable, TypeAlias

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.conformer_selection.conformer_selection import (
    ConformerSelectionResult,
)

APP_DATA_DIR = Path(__file__).parent.parent / "app_data"
CONFORMER_IMG_DIR = APP_DATA_DIR / "conformer_selection" / "img"
ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, ConformerSelectionResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data_scores = []
    for model_name, results in data.items():
        if model_name in selected_models:
            model_data_converted = {
                "Score": results.score,
                "RMSE": results.avg_rmse,
                "MAE": results.avg_mae,
            }
            converted_data_scores.append(model_data_converted)

    return pd.DataFrame(converted_data_scores, index=selected_models)


def conformer_selection_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the conformer selection benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Conformer selection")
    st.sidebar.markdown("# Conformer selection")

    st.markdown(
        "Organic molecules are flexible and able to adopt multiple conformations. "
        "These differ in energy due to strain and subtle changes in intramolecular "
        "atomic interactions. This benchmark tests the ability of MLIPs to select "
        "the most stable conformers out of an ensemble and predict the relative "
        "energy differences. The key metrics of the benchmark are the MAE and RMSE. "
        "A model that performs well on this benchmark, i.e. with low RMSE and MAE, "
        "should be able to select the most stable conformers out of an ensemble."
    )

    st.markdown(
        "This benchmark uses the Wiggle 150 dataset of highly strained conformers. "
        "The dataset contains 50 conformers each of three molecules: Adenosine, "
        "Benzylpenicillin, and Efavirenz (structures below). The benchmark runs energy "
        "inference on each of these conformers and reports the MAE and RMSE compared "
        "to the QM reference data."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit-open/benchmarks/small_molecules/conformer_selection.html)."
    )

    col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
    with col1:
        st.image(CONFORMER_IMG_DIR / "Adenosin.png", caption="Adenosine")
    with col2:
        st.image(CONFORMER_IMG_DIR / "Benzylpenicillin.png", caption="Benzylpenicillin")
    with col3:
        st.image(CONFORMER_IMG_DIR / "Efavirenz.png", caption="Efavirenz")

    st.markdown("")
    st.markdown("## Summary statistics")
    st.markdown("")

    # Download data and get model names
    if "conformer_selection_cached_data" not in st.session_state:
        st.session_state.conformer_selection_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = (
        st.session_state.conformer_selection_cached_data
    )

    model_names = list(data.keys())
    model_select = st.sidebar.multiselect(
        "Select model(s)", model_names, default=model_names
    )
    selected_models = model_select if model_select else model_names

    df = _process_data_into_dataframe(data, selected_models)
    df_display = df.copy()
    df_display.index.name = "Model Name"
    df_display = df_display.sort_values("Score", ascending=False).style.format(
        precision=3
    )

    st.dataframe(df_display)

    st.markdown("## MAE and RMSE per model")
    st.markdown("")

    # Melt the dataframe to prepare for Altair chart
    chart_df = (
        df.reset_index()
        .melt(
            id_vars=["index"],
            value_vars=["RMSE", "MAE"],
            var_name="Metric",
            value_name="Value",
        )
        .rename(columns={"index": "Model"})
    )

    # Capitalize metric names for better display
    chart_df["Metric"] = chart_df["Metric"].str.upper()

    # Create grouped bar chart
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model:N", title="Model", axis=alt.Axis(labelAngle=-45, labelLimit=100)
            ),
            y=alt.Y("Value:Q", title="Error (kcal/mol)"),
            color=alt.Color("Metric:N", title="Metric"),
            xOffset="Metric:N",
        )
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)
