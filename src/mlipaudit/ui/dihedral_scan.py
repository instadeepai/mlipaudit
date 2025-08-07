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
from ase import units

from mlipaudit.dihedral_scan.dihedral_scan import (
    DihedralScanFragmentResult,
    DihedralScanResult,
)

APP_DATA_DIR = Path.cwd() / "app_data"
DIHEDRAL_SCAN_DATA_DIR = APP_DATA_DIR / "dihedral_scan"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, DihedralScanResult]


def get_structure_data(
    data: BenchmarkResultForMultipleModels, structure_name
) -> dict[ModelName, DihedralScanFragmentResult]:
    """Get the data per model for a given structure.

    Args:
        data: The result from the benchmark.
        structure_name: The name of the structure.

    Returns:
        A dictionary of model names to fragment result
            where the fragment is the one corresponding to
            the structure name.
    """
    structure_by_model = {}
    for model_name, result in data.items():
        for fragment in result.fragments:
            if fragment.fragment_name == structure_name:
                structure_by_model[model_name] = fragment
    return structure_by_model


@st.cache_data
def load_torsion_net_data() -> dict:
    """Load the torsion net data from the data directory.

    Returns:
        A dictionary with keys that are fragment names
            to values that are dicts containing the
            corresponding torsion net data.
    """
    with open(
        DIHEDRAL_SCAN_DATA_DIR / "TorsionNet500_nocoord.json",
        "r",
        encoding="utf-8",
    ) as f:
        torsion_net_data = json.load(f)
        return torsion_net_data


def dihedral_scan_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the dihedral scan.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Dihedral scan")
    st.sidebar.markdown("# Dihedral scan")

    with st.sidebar.container():
        selected_energy_unit = st.selectbox(
            "Select an energy unit:",
            ["kcal/mol", "eV"],
        )

    st.markdown(
        "Dihedral scans are a common technique in quantum chemistry to study the "
        "effect of bond angles on the energy of a molecule. Here, we assess the "
        "ability of MLIPs to predict the energy profile of a molecule as a function "
        "of the dihedral angle of a rotatable bond in a small molecule."
    )

    st.markdown(
        "We use the TorsionNet 500 test set for this benchmark, which contains 500 "
        "structures of drug-like molecules and their energy profiles around "
        "selected rotatable bonds. The key metric of the benchmark is the average "
        "error of the barrier heights throughtout the dataset, which should be as "
        "small as possible. The barrier height is defined as the maximum of the "
        "energy profile minus the minimum. For further investingation on the model "
        "performance, the deviation of the energy profiles from the reference is "
        "computed based on RMSE and Pearson correlation. A small RMSE and high "
        "Pearson correlation indicates an energy profile that is similar to the "
        "reference."
    )

    st.markdown(
        "For more information, see the"
        " [docs](https://mlipaudit-dot-int-research-tpu.uc.r.appspot.com/benchmarks/small-molecules/dihedral_scan.html)."
    )

    # Download data and get model names
    if "cached_data" not in st.session_state:
        st.session_state.cached_data = data_func()

    # Retrieve the data from the session state
    data = st.session_state.cached_data

    unique_model_names = list(set(data.keys()))
    model_select = st.sidebar.multiselect(
        "Select model(s)", unique_model_names, default=unique_model_names
    )
    selected_models = model_select if model_select else unique_model_names

    conversion_factor = (
        1.0 if selected_energy_unit == "kcal/mol" else (units.kcal / units.mol)
    )
    score_data = [
        {
            "Model name": model_name,
            "MAE": result.avg_mae * conversion_factor,
            "RMSE": result.avg_rmse * conversion_factor,
            "Barrier Height Error": result.avg_barrier_height_error * conversion_factor,
            "Pearson Correlation": result.avg_pearson_r,
        }
        for model_name, result in data.items()
    ]

    # Create summary dataframe
    df = pd.DataFrame(score_data)

    st.markdown("## Best model summary")

    # Get best model
    best_model_row = df.loc[df["Barrier Height Error"].idxmin()]
    best_model_name = best_model_row["Model name"]

    st.markdown(
        f"The best model is **{best_model_name}** "
        f"based on minimum barrier height error."
    )
    metric_names = ["MAE", "RMSE", "Barrier Height Error", "Pearson Correlation"]
    cols_metrics = st.columns(len(metric_names))
    for i, metric_name in enumerate(metric_names):
        with cols_metrics[i]:
            st.metric(metric_name, f"{float(best_model_row[metric_name]):.3f}")

    st.markdown("## Summary statistics")
    st.markdown(
        "This table gives an overview of average error metrics for the MLIP "
        "predicted energy profiles compared to reference data. The MAE and RMSE "
        "should be as low as possible, while the Pearson correlation should be as "
        "high as possible."
    )

    st.dataframe(df, hide_index=True)

    st.markdown("## Mean barrier height error")
    df_barrier = df[df["Model name"].isin(selected_models)][
        ["Model name", "Barrier Height Error"]
    ]

    barrier_chart = (
        alt.Chart(df_barrier)
        .mark_bar()
        .encode(
            x=alt.X("Model name:N", title="Model ID"),
            y=alt.Y(
                "Barrier Height Error:Q",
                title=f"Mean Barrier Height Error ({selected_energy_unit})",
            ),
            color=alt.Color("Model name:N", title="Model ID"),
            tooltip=["Model name:N", "Barrier Height Error:Q"],
        )
        .properties(
            width=600,
            height=400,
        )
    )

    st.altair_chart(barrier_chart, use_container_width=True)
    buffer = io.BytesIO()
    barrier_chart.save(buffer, format="png", ppi=600)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="dihedral_scan_barrier_chart.png",
    )

    st.markdown("## Energy profiles")
    st.markdown(
        "Here you can skip through the energy profiles produced by the selected models "
        "for all structures in the test set and compare them to the reference data. "
        "The structures are sorted by the RMSE of one model, which can be chosen "
        "below. The structure with the highest RMSE, meaning the most dissimilar to "
        "the reference data, is shown first."
    )

    # Model selector
    selected_model_for_sorting = st.selectbox(
        "Select model for sorting by RMSE",
        selected_models,
        key="model_selector_for_sorting",
    )

    structure_rmse_structure_list = []
    for fragment in data[selected_model_for_sorting].fragments:
        structure_rmse_structure_list.append((fragment.rmse, fragment.fragment_name))

    sorted_rmse_list = sorted(structure_rmse_structure_list, reverse=True)
    sorted_structure_names = [name for rmse, name in sorted_rmse_list]

    # Initialize session state for current structure index
    if "current_structure_index" not in st.session_state:
        st.session_state.current_structure_index = 0

    # Navigation controls
    # Number input for direct navigation
    structure_number = st.number_input(
        "Structure number (sorted by descending RMSE)",
        min_value=1,
        max_value=len(sorted_structure_names),
        value=st.session_state.current_structure_index + 1,
        key="structure_number_input",
    )

    # Update index if number changed
    if structure_number - 1 != st.session_state.current_structure_index:
        st.session_state.current_structure_index = structure_number - 1
        st.rerun()

    # Get current structure data
    if sorted_structure_names:
        current_structure_name = sorted_structure_names[
            st.session_state.current_structure_index
        ]
        current_structure_data = get_structure_data(data, current_structure_name)

        # Display structure image
        image_path = DIHEDRAL_SCAN_DATA_DIR / "img" / f"{current_structure_name}.png"
        image_path = image_path.resolve()
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(image_path))

        # Create plot data for all selected models
        plot_data = []

        torsion_net_data = load_torsion_net_data()

        # Add reference data if available
        if current_structure_name in torsion_net_data:
            reference_profile = torsion_net_data[current_structure_name][
                "dft_energy_profile"
            ]
            x_values = [-180 + i * 15 for i in range(len(reference_profile))]

            for i, (x_val, energy_list) in enumerate(zip(x_values, reference_profile)):
                # Extract second element (index 1) from inner list and convert to float
                energy_val = float(energy_list[1]) * conversion_factor
                plot_data.append({
                    "model": "Reference",
                    "dihedral_angle": x_val,
                    "energy": energy_val,
                    "point_index": i,
                })

        for model_name in selected_models:
            fragment_for_model = current_structure_data[model_name]

            energy_profile = fragment_for_model.predicted_energy_profile

            # Create x-axis values starting from -180 with steps of 15
            x_values = [-180 + i * 15 for i in range(len(energy_profile))]

            for i, (x_val, energy_val) in enumerate(zip(x_values, energy_profile)):
                if isinstance(energy_val, (list, np.ndarray)):
                    processed_energy = energy_val[0] if len(energy_val) > 0 else 0.0
                else:
                    processed_energy = energy_val

                processed_energy = float(processed_energy) * conversion_factor

                plot_data.append({
                    "model": str(model_name),
                    "dihedral_angle": x_val,
                    "energy": float(processed_energy),
                    "point_index": i,
                })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # Create the energy profile chart
            energy_chart = (
                alt.Chart(plot_df)
                .mark_line(
                    point=alt.OverlayMarkDef(size=50, filled=False, strokeWidth=2)
                )
                .encode(
                    x=alt.X(
                        "dihedral_angle:Q",
                        title="Dihedral Angle (degrees)",
                        scale=alt.Scale(domain=[-180, 180]),
                    ),
                    y=alt.Y("energy:Q", title=f"Energy ({selected_energy_unit})"),
                    color=alt.Color("model:N", title="Model"),
                    tooltip=["model:N", "dihedral_angle:Q", "energy:Q"],
                )
                .interactive()
                .properties(
                    title="Energy Profile along dihedral angle", width=800, height=400
                )
            )

            st.altair_chart(energy_chart, use_container_width=True)
            buffer = io.BytesIO()
            energy_chart.save(buffer, format="png", ppi=600)
            img_bytes = buffer.getvalue()
            st.download_button(
                label="Download plot",
                data=img_bytes,
                file_name="dihedral_scan_energy_profile.png",
            )
        else:
            st.write("No energy profile data available for selected models.")
    else:
        st.write("No structure data available.")
