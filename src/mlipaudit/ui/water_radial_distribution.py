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
from pathlib import Path
from typing import Callable, TypeAlias

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.npyio import NpzFile

from mlipaudit.water_radial_distribution.water_radial_distribution import (
    WaterRadialDistributionResult,
)

APP_DATA_DIR = Path.cwd() / "app_data"
WATER_RADIAL_DISTRIBUTION_DATA_DIR = APP_DATA_DIR / "water_radial_distribution"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, WaterRadialDistributionResult
]


@st.cache_resource
def _load_tip3p() -> NpzFile:
    return np.load(WATER_RADIAL_DISTRIBUTION_DATA_DIR / "tip3p_500ps.npz")


@st.cache_resource
def _load_tip4p() -> NpzFile:
    return np.load(WATER_RADIAL_DISTRIBUTION_DATA_DIR / "tip4p_500ps.npz")


@st.cache_resource
def _load_reference_data() -> NpzFile:
    return np.load(WATER_RADIAL_DISTRIBUTION_DATA_DIR / "experimental_reference.npz")


def water_radial_distribution_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for both radial distribution
    benchmarks.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Radial distribution function")
    st.sidebar.markdown("# Radial distribution function")

    st.markdown(
        "The radial distribution function of water is a measure of the probability "
        "of finding a water molecule at a given distance from another water molecule. "
        "We benchmark the ability of the MLIPs to reproduce the experimental radial "
        "distribution function by running a simulation of a box of water molecules. "
        "Inspect the chart below, where you can see how closely the MLIPs match the "
        "experimental data. The three maxima correspond to the hydration shells of a "
        "water molecule. A correct representation of these hydration shells is crucial "
        "for an accurate modeling of water."
    )
    st.markdown(
        "This benchmark runs short simulations using the MLIPs and compares the "
        "resulting radial distribution functions to experimental data and to the"
        " classical water models TIP3P and TIP4P."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit-open/benchmarks/small-molecules/radial_distribution.html)."
    )

    # Download data and get model names
    if "water_planarity_cached_data" not in st.session_state:
        st.session_state.ring_planarity_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.ring_planarity_cached_data

    unique_model_names = list(set(data.keys()))
    model_select = st.sidebar.multiselect(
        "Select model(s)", unique_model_names, default=unique_model_names
    )
    selected_models = model_select if model_select else unique_model_names

    reference_data = _load_reference_data()
    classical_data_tip3p = _load_tip3p()
    classical_data_tip4p = _load_tip4p()

    rdf_data = {
        "tip3p": {
            "r": classical_data_tip3p["r_OO"],
            "rdf": classical_data_tip3p["g_OO"],
        },
        "tip4p": {
            "r": classical_data_tip4p["r_OO"],
            "rdf": classical_data_tip4p["g_OO"],
        },
        "experiment": {
            "r": reference_data["r_OO"],
            "rdf": reference_data["g_OO"],
        },
    }

    for model_name, result in data.items():
        if model_name in selected_models:
            rdf_data[model_name] = {
                "r": np.array(result.radii),
                "rdf": np.array(result.rdf),
            }

    score_data = {}

    # Calculate RMSD for each model with respect to experimental reference
    exp_r = rdf_data["experiment"]["r"]
    exp_rdf = rdf_data["experiment"]["rdf"]

    # Filter experimental data to range 2.5 < r < 10
    exp_mask = (exp_r > 2.5) & (exp_r < 10)
    exp_r_filtered = exp_r[exp_mask]
    exp_rdf_filtered = exp_rdf[exp_mask]

    for model_name, model_data in rdf_data.items():
        if model_name != "experiment":
            model_r = model_data["r"]
            model_rdf = model_data["rdf"]

            # Filter model data to range 2.5 < r < 10
            model_mask = (model_r > 2.5) & (model_r < 10)
            model_r_filtered = model_r[model_mask]
            model_rdf_filtered = model_rdf[model_mask]

            # Interpolate to common r grid (use experimental grid as reference)
            model_rdf_interp = np.interp(
                exp_r_filtered, model_r_filtered, model_rdf_filtered
            )

            # Calculate RMSD
            rmsd = np.sqrt(np.mean((model_rdf_interp - exp_rdf_filtered) ** 2))
            score_data[model_name] = rmsd

    score_df = pd.DataFrame([
        {"Model name": str(model), "RMSE": rmsd} for model, rmsd in score_data.items()
    ])

    st.markdown("## Best model summary")
    df_metrics = score_df.set_index("Model name")
    best_model_id = df_metrics["RMSE"].idxmin()
    st.write(f"The best model is **{best_model_id}** based on RMSE.")

    cols_metrics = st.columns(len(df_metrics.columns))
    for i, col in enumerate(df_metrics.columns):
        with cols_metrics[i]:
            st.metric(
                f"{col.replace('_', ' ')}", f"{df_metrics.loc[best_model_id, col]:.3f}"
            )

    st.markdown("## RMSE per model")
    st.markdown(
        "Here we show how much the radial distribution functions of the models "
        "deviate from the experimental data. The lower the RMSE, the better the "
        "model is at reproducing the experimental data. Only the RDF in the range "
        "between 2.5 Å and 10 Å is considered."
    )
    st.markdown("")

    bar_chart = (
        alt.Chart(score_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model name:N",
                title="Model name",
                sort=alt.EncodingSortField(field="RMSE", order="ascending"),
            ),
            y=alt.Y("RMSE:Q", title="RMSE"),
            color=alt.Color("Model name:N", title="Model name"),
        )
        .properties(width=600, height=300)
    )

    st.altair_chart(bar_chart, use_container_width=True)
    buffer = io.BytesIO()
    bar_chart.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="water_radial_distribution_chart.png",
    )

    # Create list of all available models for plotting
    all_plot_models = [str(x) for x in rdf_data.keys()]
    plot_model_display = [
        "Experimental" if model == "experiment" else model for model in all_plot_models
    ]

    # Add plot model selection interface
    plot_model_select = st.sidebar.multiselect(
        "Select models for line plot",
        plot_model_display,
        default=plot_model_display,
    )

    # Convert back to internal model names
    selected_plot_models = []
    for display_name in plot_model_select:
        if display_name == "Experimental":
            selected_plot_models.append("experiment")
        else:
            selected_plot_models.append(display_name)

    # Convert to long format for Altair plotting
    plot_data = []
    for model_name, model_data in rdf_data.items():
        # Only include models selected for plotting
        if model_name in selected_plot_models:
            r_values = model_data["r"]
            rdf_values = model_data["rdf"]
            for r_val, rdf_val in zip(r_values, rdf_values):
                # Only include data points where r < 12
                if r_val < 12:
                    plot_data.append({
                        "r": r_val,
                        "rdf": rdf_val,
                        "model": "Experimental"
                        if model_name == "experiment"
                        else str(model_name),
                    })

    df_plot = pd.DataFrame(plot_data)

    st.markdown("## Water radial distribution function")
    st.markdown(
        "Here we show the radial distribution functions of the models, as well as "
        "the experimental data. The models should be able to reproduce the minima and "
        "maxima of the three hydration shells correctly. Note that very high spikes "
        "or a flat line likely mean that the simulation of the water box has exploded."
    )
    st.markdown("")

    # Create Altair line chart
    chart = (
        alt.Chart(df_plot)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("r:Q", title="Distance r (Å)"),
            y=alt.Y("rdf:Q", title="O-O Radial Distribution Function"),
            color=alt.Color("model:N", title="Model name"),
        )
        .properties(width=800, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    buffer = io.BytesIO()
    chart.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="water_radial_distribution_chart.png",
        key="water_radial_distribution_chart",
    )
