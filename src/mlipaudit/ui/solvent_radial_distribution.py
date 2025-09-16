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
import json
from pathlib import Path
from typing import Callable, TypeAlias

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.npyio import NpzFile

from mlipaudit.solvent_radial_distribution import (
    SolventRadialDistributionResult,
)

APP_DATA_DIR = Path(__file__).parent.parent / "app_data"
SOLVENT_RADIAL_DISTRIBUTION_DATA_DIR = APP_DATA_DIR / "solvent_radial_distribution"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, SolventRadialDistributionResult
]


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
    st.sidebar.markdown("# Solvent Radial distribution function")

    st.markdown(
        "Here we show the radial distribution function of the solvents CCl4, "
        "methanol, and acetonitrile. The vertical lines show the reference "
        "maximum of the radial distribution function for each solvent."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit-open/benchmarks/small-molecules/radial_distribution.html)."
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

    unique_model_names = list(set(data.keys()))
    model_select = st.sidebar.multiselect(
        "Select model(s)", unique_model_names, default=unique_model_names
    )
    selected_models = model_select if model_select else unique_model_names

    solvent_maxima = _load_experimental_data()

    solvent_data = [
        {"Model name": model_name, "Avg peak deviation": result.avg_peak_deviation}
        for model_name, result in data.items()
        if model_name in selected_models
    ]
    df = pd.DataFrame(solvent_data)

    st.markdown("## Best model summary")

    # Get best model
    best_model_row = df.loc[df["Avg peak deviation"].idxmin()]
    best_model_name = best_model_row["Model name"]

    st.markdown(
        f"The best model is **{best_model_name}** based on average peak deviation."
    )

    cols_metrics = st.columns(4)
    with cols_metrics[0]:
        st.metric(
            "Average peak deviation",
            f"{float(best_model_row['Avg peak deviation']):.3f}",
        )
    for solvent_index, solvent_name in enumerate(data[best_model_name].structure_names):
        with cols_metrics[solvent_index + 1]:
            deviation = data[best_model_name].structures[solvent_index].peak_deviation
            st.metric(
                solvent_name,
                f"{float(deviation):.3f}",
            )

    st.markdown("## Radial distribution functions")

    for solvent_index, solvent in enumerate(["CCl4", "methanol", "acetonitrile"]):
        rdf_data_solvent = {}

        for model_name, result in data.items():
            if model_name in selected_models and solvent in result.structure_names:
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
                    if r_val < 12:
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
                    color=alt.Color("model:N", title="Model ID"),
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

            # Add download button for solvent plot
            buffer_solvent = io.BytesIO()
            combined_chart.save(buffer_solvent, format="png", ppi=300)
            img_bytes_solvent = buffer_solvent.getvalue()
            st.download_button(
                label="Download plot",
                data=img_bytes_solvent,
                file_name=f"{solvent}_radial_distribution_chart.png",
                key=f"{solvent}_radial_distribution_chart",
            )
        else:
            st.warning(f"No data found for {solvent}")
