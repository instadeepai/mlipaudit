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

import io
from typing import Callable, TypeAlias

import altair as alt
import pandas as pd
import streamlit as st
from ase import units

from mlipaudit.tautomers.tautomers import TautomersResult

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
    st.sidebar.markdown("# Tautomers")

    st.markdown(
        "Tautomers are isomers that can interconvert by the movement "
        "of a proton and/or the rearrangement of double bonds. "
        "This benchmark evaluates "
        "how well MLIPs can predict the relative energies and stability of different "
        "tautomeric forms of molecules in-vacuum. "
        "The dataset contains 1391 tautomer pairs with reference QM energies extracted "
        "from the Tautobase dataset."
    )

    st.markdown(
        "For more information, see the [docs](https://mlipaudit-dot-int-research-"
        "tpu.uc.r.appspot.com/benchmarks/small-molecules/tautomers.html)."
    )

    with st.sidebar.container():
        unit_selection = st.selectbox(
            "Select an energy unit:",
            ["kcal/mol", "eV"],
        )

    # Set conversion factor based on selection
    if unit_selection == "kcal/mol":
        conversion_factor = 1.0
        unit_label = "kcal/mol"
    else:
        conversion_factor = units.kcal / units.mol
        unit_label = "eV"

    # Download data and get model names
    if "cached_data" not in st.session_state:
        st.session_state.cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.cached_data

    unique_model_names = list(set(data.keys()))

    # Add "Select All" option
    all_models_option = st.sidebar.checkbox("Select all models", value=False)

    if all_models_option:
        model_select = unique_model_names
    else:
        model_select = st.sidebar.multiselect(
            "Select models", unique_model_names, default=unique_model_names
        )

    selected_models = model_select if model_select else unique_model_names

    # Convert to long-format DataFrame
    converted_data = []
    for model_name, model_data in data.items():
        for molecule in model_data.molecules:
            converted_data.append({
                "model": model_name,
                "structure ID": molecule.structure_id,
                "abs_deviation": molecule.abs_deviation * conversion_factor,
                "pred_energy_diff": molecule.predicted_energy_diff * conversion_factor,
                "ref_energy_diff": molecule.ref_energy_diff * conversion_factor,
            })

    df_detailed = pd.DataFrame(converted_data)

    # Calculate MAE and RMSE for each model_id
    metrics_data = []
    for model_name in selected_models:
        mae = data[model_name].mae * conversion_factor
        rmse = data[model_name].rmse * conversion_factor
        metrics_data.extend([
            {"model": model_name, "metric": "MAE", "value": mae},
            {"model": model_name, "metric": "RMSE", "value": rmse},
        ])

    metrics_df = pd.DataFrame(metrics_data)

    st.markdown("## Best model summary")

    # Find model with lowest RMSE
    rmse_data = metrics_df[metrics_df["metric"] == "RMSE"]
    best_idx = rmse_data["value"].idxmin()
    best_model_name = rmse_data.loc[best_idx, "model"]
    st.markdown(f"The best model is **{best_model_name}** based on RMSE.")

    cols_metrics = st.columns(2)
    for i, metric in enumerate(["MAE", "RMSE"]):
        with cols_metrics[i]:
            # Filter for specific model and metric
            filtered_df = metrics_df[
                (metrics_df["model"] == best_model_name)
                & (metrics_df["metric"] == metric)
            ]
            metric_value = filtered_df["value"].iloc[0]
            st.metric(metric, f"{float(metric_value):.3f}")

    # Create grouped bar chart
    st.markdown("## MAE and RMSE by model")
    chart = (
        alt.Chart(metrics_df)
        .mark_bar()
        .add_selection(alt.selection_interval())
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("value:Q", title=f"Error ({unit_label})"),
            color=alt.Color(
                "metric:N",
                title="Metric",
            ),
            xOffset=alt.XOffset("metric:N"),
            tooltip=["Model:N", "Metric:N", "Value:Q"],
        )
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)
    buffer = io.BytesIO()
    chart.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="tautomers_chart.png",
    )

    @st.cache_data
    def convert_for_download(df):
        return df.to_csv().encode("utf-8")

    csv = convert_for_download(df_detailed)
    st.download_button(
        label="Download full table as CSV",
        data=csv,
        file_name="tautomers_data.csv",
    )
