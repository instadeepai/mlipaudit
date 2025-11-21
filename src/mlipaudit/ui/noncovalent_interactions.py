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

from mlipaudit.benchmarks import (
    NoncovalentInteractionsBenchmark,
    NoncovalentInteractionsResult,
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
NCI_ATLAS_DIR = APP_DATA_DIR / "noncovalent_interactions"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, NoncovalentInteractionsResult
]


def _process_data_into_rmse_per_dataset(
    data: BenchmarkResultForMultipleModels,
    model_select: list[str],
    conversion_factor: float,
    subset: bool = False,
) -> pd.DataFrame:
    """Process the data into a dictionary of RMSE per subset.

    Args:
        data: The benchmark results.
        model_select: The models to include in the DataFrame.
        conversion_factor: The conversion factor for the energy unit.
        subset: Whether to process into RMSE per dataset or per subset.

    Returns:
        A pandas DataFrame with the RMSE per subset or dataset.
    """
    converted_data = []
    model_names = []
    for model_name, result in data.items():
        if model_name in model_select:
            row_data = {"Score": result.score}
            if subset:
                row_data.update(**result.rmse_interaction_energy_subsets)
            else:
                row_data.update(**result.rmse_interaction_energy_datasets)
            converted_data.append(row_data)
            model_names.append(model_name)

    df = pd.DataFrame(converted_data, index=model_names)
    df = df.dropna(axis=1, how="all")
    columns_to_convert = [col for col in df.columns if col != "Score"]
    df[columns_to_convert] = df[columns_to_convert] * conversion_factor
    return df


def _get_energy_profiles_for_subset(
    data: BenchmarkResultForMultipleModels,
    subset: str,
    model_select: list[str],
    conversion_factor: float,
) -> dict[ModelName, dict[str, tuple[list[float], list[float]]]]:
    """Get the energy profiles for a subset.

    Args:
        data: The benchmark results.
        subset: The subset to get the energy profiles for.
        model_select: The models to include in the DataFrame.
        conversion_factor: The conversion factor for the energy unit.

    Returns:
        A dictionary of energy profiles for the subset.
    """
    energy_profiles_per_model: dict[
        ModelName, dict[str, tuple[list[float], list[float]]]
    ] = {}
    for model_name, results in data.items():
        if model_name in model_select:
            energy_profiles_per_model[model_name] = {}
            energy_profiles_per_model["Reference"] = {}

            for system_result in results.systems:
                if system_result.failed:
                    continue

                system_subset_name = f"{system_result.dataset}: {system_result.group}"

                if system_subset_name == subset:
                    energy_profile = system_result.energy_profile
                    ref_energy_profile = system_result.reference_energy_profile
                    distance_profile = system_result.distance_profile

                    dist_idx_sorted = np.argsort(distance_profile)
                    max_dist_idx = np.argmax(distance_profile)

                    energy_profile_sorted = [
                        float(energy) * conversion_factor
                        - float(energy_profile[max_dist_idx]) * conversion_factor  # type: ignore
                        for energy in np.array(energy_profile)[dist_idx_sorted]
                    ]
                    ref_energy_profile_sorted = [
                        float(energy) * conversion_factor
                        - float(ref_energy_profile[max_dist_idx]) * conversion_factor  # type: ignore
                        for energy in np.array(ref_energy_profile)[dist_idx_sorted]
                    ]

                    energy_profiles_per_model[model_name][
                        system_result.structure_name
                    ] = (
                        sorted(distance_profile),  # type: ignore
                        energy_profile_sorted,
                    )
                    energy_profiles_per_model["Reference"][
                        system_result.structure_name
                    ] = (
                        sorted(distance_profile),  # type: ignore
                        ref_energy_profile_sorted,
                    )

    return energy_profiles_per_model


def noncovalent_interactions_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the noncovalent interactions benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Non-covalent interactions")

    st.markdown(
        "This benchmark tests if the MLIPs can reproduce interaction energies of "
        "molecular complexes driven by non-covalent interactions. The benchmark "
        "uses six datasets from the NCI Atlas: D442x10 (London dispersion), "
        "HB375x10 (hydrogen bonds), HB300SPXx10 (hydrogen bonds extended to S, P "
        "and halogens), IHB100x10 (ionic hydrogen bonds), R739x5 (repulsive "
        "contacts) and SH250x10 (sigma hole interaction). These contain QM "
        "optimized geometries of distance scans of bi-molecular complexes, where "
        "the two molecules interact via non-covalent interactions with associated "
        "energies. Each dataset contains multiple subsets which specify certain "
        "categories of interactions. The key metric is the RMSE of the interaction "
        "energies. These are defined as the bottom of the potential energy curve "
        "minus the energy of the separated molecules. For repulsive contacts, the "
        "maximum of the energy profile is used instead."
    )
    st.markdown(
        "For more information see the [docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/small_molecules/noncovalent_interactions.html)"
        " and the [NCI Atlas webpage](http://www.nciatlas.org/)."
    )
    st.markdown(
        "The benchmark skips all structures that contain elements which were completely"
        " absent from the MLIP's training data. Scroll to the end of the page to find "
        "an overview of how many structures were skipped for each dataset."
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
    if "noncov_inter_cached_data" not in st.session_state:
        st.session_state.noncov_inter_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = st.session_state.noncov_inter_cached_data

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

    df = _process_data_into_rmse_per_dataset(
        data,
        selected_models,
        conversion_factor,
        subset=False,
    )

    st.markdown("## Summary statistics")
    st.markdown(
        "This table shows the average RMSE of the interaction energies for each "
        "interaction type and model. For a more fine-grained breakdown of "
        "interaction types, see the bar plot below."
    )
    df = df.rename_axis("Model name")
    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## RMSE per data subset")
    df_subset = _process_data_into_rmse_per_dataset(
        data,
        selected_models,
        conversion_factor,
        subset=True,
    )

    # Drop the score for the rest of processing
    df_subset = df_subset.drop(columns=["Score"])

    # Reshape dataframe for Altair plotting
    df_melted = (
        df_subset.reset_index()
        .melt(id_vars=["index"], var_name="Interaction type", value_name="RMSE")
        .rename(columns={"index": "Model name"})
    )

    # Create horizontal bar plot
    selection = alt.selection_point(fields=["Model name"], bind="legend")

    chart = (
        alt.Chart(df_melted)
        .mark_bar(size=10)
        .add_params(selection)
        .encode(
            y=alt.Y(
                "Interaction type:N",
                title="Interaction Type",
                axis=alt.Axis(labelLimit=1000),
            ),
            x=alt.X("RMSE:Q", title="RMSE (kcal/mol)"),
            yOffset=alt.YOffset("Model name:N"),
            color=alt.Color("Model name:N", title="Model"),
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("Model name:N", title="Model"),
                "Interaction type:N",
                "RMSE:Q",
            ],
        )
        .resolve_scale(color="independent")
        .properties(
            width=800,
            height=max(len(df_melted) * 50, 400),
        )
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("## Energy profiles")
    st.markdown(
        "The energy profiles below show the energy of the complex as a "
        "function of the distance between the two molecules. For more "
        "information about the molecular complexes indicated by the "
        "structure names, browse the datasets on "
        "the [NCI Atlas webpage](http://www.nciatlas.org/)."
    )

    available_subsets: set[str] = set()
    for _, results in data.items():
        available_subsets.update(results.rmse_interaction_energy_subsets.keys())

    dataset_selector_set = set()
    subset_selector_set = set()
    for subset_name in available_subsets:
        dataset_selector_set.add(subset_name.split(":")[0].strip())
        subset_selector_set.add(subset_name.split(":")[1].strip())

    dataset_selector = st.selectbox(
        "Select a dataset",
        dataset_selector_set,
    )
    subset_selector = st.selectbox(
        "Select a subset",
        subset_selector_set,
    )
    # Second model selection
    model_names = list(set(data.keys()))
    model_select = st.sidebar.multiselect(
        "Select model(s)", model_names, default=model_names
    )
    model_selector_sorting = st.selectbox(
        "Select a model for sorting by interaction energy error",
        model_select,
    )

    selected_subset = f"{dataset_selector}: {subset_selector}"

    energy_profiles_per_model = _get_energy_profiles_for_subset(
        data,
        selected_subset,
        model_select,
        conversion_factor,
    )

    # Get structure names and sort them by deviation for the selected model
    if (
        model_selector_sorting in energy_profiles_per_model
        and energy_profiles_per_model[model_selector_sorting]
    ):
        structure_names = list(energy_profiles_per_model[model_selector_sorting].keys())

        # Create a mapping from structure_name to deviation for sorting
        structure_to_deviation = {}
        for system_result in data[model_selector_sorting].systems:
            if (
                f"{system_result.dataset}: {system_result.group}" == selected_subset
                and system_result.structure_name in structure_names
                and not system_result.failed
            ):
                structure_to_deviation[system_result.structure_name] = abs(
                    system_result.deviation  # type: ignore
                )

        structure_names.sort(key=lambda x: structure_to_deviation.get(x, 0))

        # Add structure selection dropdown
        selected_structure = st.selectbox(
            "Select a structure to plot",
            structure_names,
        )

        # Create DataFrame for Altair plotting
        plot_data = []
        for model_name, structure_data in energy_profiles_per_model.items():
            for structure_name, (distances, energies) in structure_data.items():
                if structure_name == selected_structure:
                    for dist, energy in zip(distances, energies):
                        plot_data.append({
                            "distance": dist,
                            "energy": energy,
                            "model": model_name,
                        })
                    break

        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            # Create the line plot
            line_chart = (
                alt.Chart(df_plot)
                .mark_line(
                    point=alt.OverlayMarkDef(size=50, filled=False, strokeWidth=2)
                )
                .encode(
                    x=alt.X("distance:Q", title="Distance (Å)"),
                    y=alt.Y("energy:Q", title=f"Energy ({selected_energy_unit})"),
                    color=alt.Color("model:N", title="Model"),
                    tooltip=["distance:Q", "energy:Q", "model:N"],
                )
                .interactive()
                .properties(width=800, height=400)
            )

            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.warning(
                "No energy profile data available for the selected subset and "
                "structure."
            )
    else:
        st.warning(
            f"No energy profile data available for model '{model_selector_sorting}' "
            "in the selected subset."
        )

    st.markdown("## Skipped and failed structures per dataset")
    st.markdown(
        "This table shows the number of structures that were skipped for each data "
        "subset and model. The first row shows the total number of structures in "
        "each data subset."
    )

    with open(
        NCI_ATLAS_DIR / "n_systems_per_subset.json",
        mode="r",
        encoding="utf-8",
    ) as f:
        n_systems_per_subset = json.load(f)

    subsets = list(n_systems_per_subset.keys())

    converted_data = []
    for model_name, results in data.items():
        if (
            len(model_select) > 0
            and model_name in model_select
            or len(model_select) == 0
        ):
            n_systems_per_subset_for_model = {}
            n_failed_per_subset_for_model = {}
            for subset in subsets:
                n_systems_per_subset_for_model[subset] = 0
                for system in results.systems:
                    subset_name_system = f"{system.dataset}: {system.group}"
                    if subset_name_system == subset:
                        n_systems_per_subset_for_model[subset] += 1

            for subset in subsets:
                # Num of failures = num of systems - num of successful systems
                n_failed_per_subset_for_model[subset] = (
                    n_systems_per_subset[subset]
                    - n_systems_per_subset_for_model[subset]
                )

            converted_data.append(n_failed_per_subset_for_model)

    df = pd.DataFrame(converted_data, index=selected_models)
    st.dataframe(df)


class NoncovalentInteractionsPageWrapper(UIPageWrapper):
    """Page wrapper for noncovalent interactions benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return noncovalent_interactions_page

    @classmethod
    def get_benchmark_class(cls) -> type[NoncovalentInteractionsBenchmark]:  # noqa: D102
        return NoncovalentInteractionsBenchmark
