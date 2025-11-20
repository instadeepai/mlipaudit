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

import statistics
from pathlib import Path
from typing import Callable, TypeAlias

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import ConformerSelectionBenchmark, ConformerSelectionResult
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
CONFORMER_IMG_DIR = APP_DATA_DIR / "conformer_selection" / "img"
ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, ConformerSelectionResult]


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data_scores = []
    model_names = []
    for model_name, results in data.items():
        if model_name in selected_models:
            model_data_converted = {
                "Score": results.score,
                "Average RMSE (kcal/mol)": results.avg_rmse,
                "Average MAE (kcal/mol)": results.avg_mae,
                "Average Spearman correlation": statistics.mean(
                    r.spearman_correlation
                    for r in results.molecules
                    if not r.failed  # type: ignore
                ),
            }
            converted_data_scores.append(model_data_converted)
            model_names.append(model_name)

    return pd.DataFrame(converted_data_scores, index=model_names)


def _molecule_stats_df(results: ConformerSelectionResult) -> pd.DataFrame:
    """Return a dataframe with per-molecule stats for a benchmark result."""
    rows = []
    for m in results.molecules:
        rows.append({
            "Molecule": m.molecule_name,
            "MAE (kcal/mol)": float(m.mae) if not m.failed else None,  # type: ignore
            "RMSE (kcal/mol)": float(m.rmse) if not m.failed else None,  # type: ignore
            "Spearman": float(m.spearman_correlation) if not m.failed else None,  # type: ignore
            "Spearman p": float(m.spearman_p_value) if not m.failed else None,  # type: ignore
        })
    df = pd.DataFrame(rows).set_index("Molecule")
    return df


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
        "[docs](https://instadeepai.github.io/mlipaudit/benchmarks/"
        "small_molecules/conformer_selection.html)."
    )

    col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
    with col1:
        create_st_image(CONFORMER_IMG_DIR / "rsz_ado00.png", "Adenosine")
    with col2:
        create_st_image(CONFORMER_IMG_DIR / "rsz_bpn00.png", "Benzylpenicillin")
    with col3:
        create_st_image(CONFORMER_IMG_DIR / "rsz_efa00.png", "Efavirenz")

    # Download data and get model names
    if "conformer_selection_cached_data" not in st.session_state:
        st.session_state.conformer_selection_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = (
        st.session_state.conformer_selection_cached_data
    )

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

    df = _process_data_into_dataframe(data, selected_models)

    st.markdown("## Summary statistics")

    df_display = df.copy()
    df_display.index.name = "Model name"
    df_display.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df_display)

    st.markdown("## MAE and RMSE per model")

    # Melt the dataframe to prepare for Altair chart
    chart_df = (
        df.reset_index()
        .melt(
            id_vars=["index"],
            value_vars=["Average RMSE (kcal/mol)", "Average MAE (kcal/mol)"],
            var_name="Metric",
            value_name="Value",
        )
        .rename(columns={"index": "Model"})
    )

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

    # inside conformer_selection_page, add after the existing chart display
    st.markdown("## Per-molecule statistics")
    st.markdown(
        "Per-molecule MAE, RMSE and Spearman correlation for each selected model."
    )

    for model_name in selected_models:
        results = data.get(model_name)
        if results is None:
            continue

        st.markdown(f"### {model_name}")
        mol_df = _molecule_stats_df(results)

        # Display table
        st.dataframe(mol_df.round(4))

        # Error chart (MAE and RMSE)
        error_chart_df = mol_df.reset_index().melt(
            id_vars=["Molecule"],
            value_vars=["MAE (kcal/mol)", "RMSE (kcal/mol)"],
            var_name="Metric",
            value_name="Value",
        )
        error_chart = (
            alt.Chart(error_chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Molecule:N",
                    title="Molecule",
                    axis=alt.Axis(labelAngle=-45, labelLimit=100),
                ),
                y=alt.Y("Value:Q", title="Error (kcal/mol)"),
                color="Metric:N",
                xOffset="Metric:N",
            )
            .properties(width=600, height=250)
        )
        st.altair_chart(error_chart, use_container_width=True)

    # Plot correlation chart for a chosen molecule and model

    # Create selectboxes for model and structure selection
    col1, col2 = st.columns(2)
    with col1:
        selected_plot_model = st.selectbox(
            "Select model for plot:", selected_models, key="model_selector_plot"
        )

    unique_structures = list(
        set([
            mol.molecule_name
            for mol in data[selected_plot_model].molecules
            if not mol.failed
        ])
    )

    with col2:
        selected_structure = st.selectbox(
            "Select structure for plot:",
            unique_structures,
            key="structure_selector_plot",
        )

    model_data_for_plot = [
        mol
        for mol in data[selected_plot_model].molecules
        if mol.molecule_name == selected_structure
    ][0]
    scatter_data = []
    for pred_energy, ref_energy in zip(
        model_data_for_plot.predicted_energy_profile,  # type: ignore
        model_data_for_plot.reference_energy_profile,  # type: ignore
    ):
        scatter_data.append({
            "Predicted Energy": pred_energy,
            "Reference Energy": ref_energy,
        })

    structure_df = pd.DataFrame(scatter_data)

    spearman_corr = structure_df["Predicted Energy"].corr(
        structure_df["Reference Energy"], method="spearman"
    )

    # Create scatter plot
    scatter_chart = (
        alt.Chart(structure_df)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("Reference Energy:Q", title="Reference Energy (kcal/mol)"),
            y=alt.Y("Predicted Energy:Q", title="Predicted Energy (kcal/mol)"),
            tooltip=["Reference Energy:Q", "Reference Energy:Q"],
        )
        .properties(
            width=600,
            height=400,
            title=(
                f"Model {selected_plot_model} - {selected_structure} "
                f"(Spearman ρ = {spearman_corr:.3f})"
            ),
        )
    )

    # Add diagonal line for perfect correlation
    min_energy = min(
        structure_df["Reference Energy"].min(), structure_df["Predicted Energy"].min()
    )
    max_energy = max(
        structure_df["Reference Energy"].max(), structure_df["Predicted Energy"].max()
    )

    diagonal_line = (
        alt.Chart(
            pd.DataFrame({"x": [min_energy, max_energy], "y": [min_energy, max_energy]})
        )
        .mark_line(color="gray", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )

    # Combine scatter plot and diagonal line
    final_chart = scatter_chart + diagonal_line

    st.altair_chart(final_chart, use_container_width=True)


class ConformerSelectionPageWrapper(UIPageWrapper):
    """Page wrapper for conformer selection benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return conformer_selection_page

    @classmethod
    def get_benchmark_class(cls) -> type[ConformerSelectionBenchmark]:  # noqa: D102
        return ConformerSelectionBenchmark
