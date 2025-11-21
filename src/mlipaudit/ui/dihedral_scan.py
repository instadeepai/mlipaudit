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
from ase import units

from mlipaudit.benchmarks.dihedral_scan.dihedral_scan import (
    DihedralScanBenchmark,
    DihedralScanFragmentResult,
    DihedralScanResult,
)
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    create_st_image,
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

APP_DATA_DIR = Path(__file__).parent.parent / "app_data"
DIHEDRAL_SCAN_DATA_DIR = APP_DATA_DIR / "dihedral_scan"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, DihedralScanResult]


def get_structure_data(
    data: BenchmarkResultForMultipleModels, structure_name
) -> dict[ModelName, DihedralScanFragmentResult]:
    """Get the data per model for a given structure.
    Don't add fragments that failed.

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
            if not fragment.failed and fragment.fragment_name == structure_name:
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
        "We use the [TorsionNet 500](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01346)"
        " test set for this benchmark, which contains 500 "
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
        " [docs](https://instadeepai.github.io/mlipaudit/benchmarks"
        "/small_molecules/dihedral_scan.html)."
    )

    # Download data and get model names
    if "dihedral_scan_cached_data" not in st.session_state:
        st.session_state.dihedral_scan_cached_data = data_func()

    # Retrieve the data from the session state
    data = st.session_state.dihedral_scan_cached_data

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

    conversion_factor = (
        1.0 if selected_energy_unit == "kcal/mol" else (units.kcal / units.mol)
    )
    score_data = [
        {
            "Model name": model_name,
            "Score": result.score,
            f"MAE ({selected_energy_unit})": result.avg_mae * conversion_factor,
            f"RMSE ({selected_energy_unit})": result.avg_rmse * conversion_factor,
            f"Barrier Height MAE ({selected_energy_unit})": result.mae_barrier_height
            * conversion_factor,
            "Pearson Correlation": result.avg_pearson_r,
        }
        for model_name, result in data.items()
        if model_name in selected_models
    ]

    # Create summary dataframe
    df = pd.DataFrame(score_data)

    st.markdown("## Summary statistics")
    st.markdown(
        "This table gives an overview of average error metrics for the MLIP "
        "predicted energy profiles compared to reference data. The MAE and RMSE "
        "should be as low as possible, while the Pearson correlation should be as "
        "high as possible."
    )
    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## Mean barrier height error")
    df_barrier = df[df["Model name"].isin(selected_models)][
        ["Model name", f"Barrier Height MAE ({selected_energy_unit})"]
    ]

    barrier_chart = (
        alt.Chart(df_barrier)
        .mark_bar()
        .encode(
            x=alt.X(
                "Model name:N",
                title="Model",
                axis=alt.Axis(labelAngle=-45, labelLimit=100),
            ),
            y=alt.Y(
                f"Barrier Height MAE ({selected_energy_unit}):Q",
                title=f"Mean Barrier Height Error ({selected_energy_unit})",
            ),
            color=alt.Color("Model name:N", title="Model"),
            tooltip=[
                alt.Tooltip("Model name:N", title="Model"),
                f"Barrier Height MAE ({selected_energy_unit}):Q",
            ],
        )
        .properties(
            width=600,
            height=400,
        )
    )

    st.altair_chart(barrier_chart, use_container_width=True)

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
        # Don't allow choosing a failed fragment
        if fragment.failed:
            continue

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
            create_st_image(image_path)

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
            # Skip the model if its respective fragment failed
            if model_name not in current_structure_data:
                continue

            fragment_for_model = current_structure_data[model_name]

            energy_profile = fragment_for_model.predicted_energy_profile

            # Create x-axis values starting from -180 with steps of 15
            x_values = [-180 + i * 15 for i in range(len(energy_profile))]  # type: ignore

            for i, (x_val, energy_val) in enumerate(zip(x_values, energy_profile)):  # type: ignore
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
        else:
            st.write("No energy profile data available for selected models.")
    else:
        st.write("No structure data available.")


class DihedralScanPageWrapper(UIPageWrapper):
    """Page wrapper for dihedral scan page."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return dihedral_scan_page

    @classmethod
    def get_benchmark_class(cls) -> type[DihedralScanBenchmark]:  # noqa: D102
        return DihedralScanBenchmark
