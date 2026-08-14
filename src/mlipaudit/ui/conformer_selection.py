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

MAE_COLUMN = "MAE (kcal/mol)"
RMSE_COLUMN = "RMSE (kcal/mol)"
SPEARMAN_COLUMN = "Spearman"
PER_MOLECULE_METRICS = [MAE_COLUMN, RMSE_COLUMN, SPEARMAN_COLUMN]
DEFAULT_NUM_WORST_MOLECULES = 20


def _mean_or_none(values: list[float]) -> float | None:
    """Average a list of metric values, tolerating an empty list.

    Args:
        values: The metric values to average.

    Returns:
        The mean of the values, or None if there are no values.
    """
    return statistics.mean(values) if values else None


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    converted_data_scores = []
    model_names = []
    for model_name, results in data.items():
        if model_name in selected_models:
            successful = [m for m in results.molecules if not m.failed]
            model_data_converted = {
                "Score": results.score,
                "Average RMSE (kcal/mol)": results.avg_rmse,
                "Average MAE (kcal/mol)": results.avg_mae,
                "Average Spearman correlation": _mean_or_none([
                    m.spearman_correlation
                    for m in successful
                    if m.spearman_correlation is not None
                ]),
                "Molecules evaluated": len(successful),
                "Molecules failed": len(results.molecules) - len(successful),
            }
            converted_data_scores.append(model_data_converted)
            model_names.append(model_name)

    return pd.DataFrame(converted_data_scores, index=model_names)


def _per_molecule_df(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    """Return a long-format dataframe with per-molecule stats for all models.

    Args:
        data: The benchmark results per model.
        selected_models: The models to include.

    Returns:
        A dataframe with one row per model and molecule.
    """
    rows = []
    for model_name in selected_models:
        results = data.get(model_name)
        if results is None:
            continue
        for m in results.molecules:
            rows.append({
                "Model": model_name,
                "Molecule": m.molecule_name,
                MAE_COLUMN: float(m.mae) if m.mae is not None else None,
                RMSE_COLUMN: float(m.rmse) if m.rmse is not None else None,
                SPEARMAN_COLUMN: (
                    float(m.spearman_correlation)
                    if m.spearman_correlation is not None
                    else None
                ),
                "Spearman p": (
                    float(m.spearman_p_value)
                    if m.spearman_p_value is not None
                    else None
                ),
                "Failed": m.failed,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Molecule",
            MAE_COLUMN,
            RMSE_COLUMN,
            SPEARMAN_COLUMN,
            "Spearman p",
            "Failed",
        ],
    )


def _error_distribution_chart(molecule_df: pd.DataFrame, metric: str) -> alt.Chart:
    """Build a per-molecule metric distribution histogram layered over models.

    The dataset contains hundreds of molecules, so distributions are shown instead
    of one bar per molecule.

    Args:
        molecule_df: The long-format per-molecule dataframe.
        metric: The column to histogram.

    Returns:
        The Altair chart.
    """
    return (
        alt.Chart(molecule_df.dropna(subset=[metric]))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=50), title=metric),
            y=alt.Y("count():Q", title="Number of molecules", stack=None),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("count():Q", title="Molecules"),
            ],
        )
        .properties(width=600, height=350)
    )


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
        "This benchmark uses the Folmsbee dataset, which contains up to 10 "
        "near-minimum conformers for each of around 700 organic molecules. The "
        "reference level of theory for the energy labels is DLPNO-CCSD(T). The "
        "benchmark runs energy inference on all conformers of a molecule and, after "
        "aligning both energy profiles to the lowest-energy reference conformer, "
        "reports the MAE, RMSE and Spearman rank correlation of that molecule's "
        "energy profile. The reported averages are taken over all molecules. Below "
        'are the conformer ensembles of two example molecules, "astex_1gkc" and '
        '"omegacsd_HEKZAY".'
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit/benchmarks/"
        "small_molecules/conformer_selection.html)."
    )

    col1, col2 = st.columns(2, vertical_alignment="bottom")
    with col1:
        create_st_image(CONFORMER_IMG_DIR / "rsz_astex_1gkc.png", "astex_1gkc")
    with col2:
        create_st_image(
            CONFORMER_IMG_DIR / "rsz_omegacsd_HEKZAY.png", "omegacsd_HEKZAY"
        )

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

    selected_models = [name for name in selected_models if name in data]
    if not selected_models:
        st.markdown("**No results to display**.")
        return

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

    st.altair_chart(chart, width="stretch")

    molecule_df = _per_molecule_df(data, selected_models)

    st.markdown("## Distribution of per-molecule metrics")
    st.markdown(
        "The dataset contains too many molecules to show them individually, so the "
        "histogram below shows how a per-molecule metric is distributed over all "
        "molecules for each selected model. A model with a narrow distribution close "
        "to zero error (or close to a Spearman correlation of one) performs well "
        "across the whole dataset."
    )

    selected_metric = st.selectbox(
        "Select a metric:",
        PER_MOLECULE_METRICS,
        key="conformer_metric_selector",
    )
    st.altair_chart(
        _error_distribution_chart(molecule_df, selected_metric),
        width="stretch",
    )

    st.markdown("## Per-molecule statistics")
    st.markdown(
        "Per-molecule MAE, RMSE and Spearman correlation for a selected model, "
        "ordered by the metric selected above, worst molecules first. Molecules on "
        "which the inference failed have no metrics and are listed at the end."
    )

    selected_table_model = st.selectbox(
        "Select a model:",
        sorted(selected_models),
        key="conformer_table_model_selector",
    )
    model_molecule_df = (
        molecule_df[molecule_df["Model"] == selected_table_model]
        .drop(columns=["Model"])
        .set_index("Molecule")
    )
    # Sorting by the selected metric puts the largest errors first, and the lowest
    # Spearman correlations first, i.e. the molecules the model handles worst.
    sort_ascending = selected_metric == SPEARMAN_COLUMN
    model_molecule_df = model_molecule_df.sort_values(
        selected_metric, ascending=sort_ascending, na_position="last"
    )

    num_molecules = len(model_molecule_df)
    num_to_show = num_molecules
    if num_molecules > DEFAULT_NUM_WORST_MOLECULES:
        num_to_show = st.slider(
            "Number of molecules to show:",
            min_value=5,
            max_value=num_molecules,
            value=DEFAULT_NUM_WORST_MOLECULES,
            key="conformer_num_molecules_slider",
        )

    st.dataframe(model_molecule_df.head(num_to_show).round(4))
    st.markdown(
        f"Showing {min(num_to_show, num_molecules)} out of {num_molecules} molecules."
    )

    # Plot correlation chart for a chosen molecule and model
    st.markdown("## Conformer energy profiles")
    st.markdown(
        "Predicted against reference conformer energies for a single molecule, both "
        "relative to the lowest-energy reference conformer. Points on the dashed "
        "diagonal are perfectly predicted."
    )

    # Create selectboxes for model and structure selection
    col1, col2 = st.columns(2)
    with col1:
        selected_plot_model = st.selectbox(
            "Select model for plot:", sorted(selected_models), key="model_selector_plot"
        )

    # Molecules can be ordered by name, or by error so that the worst cases for the
    # selected model are easy to find among the many molecules of the dataset.
    plot_molecule_df = molecule_df[
        (molecule_df["Model"] == selected_plot_model) & (~molecule_df["Failed"])
    ]
    worst_first_option = (
        f"Lowest {selected_metric} first"
        if selected_metric == SPEARMAN_COLUMN
        else f"Largest {selected_metric} first"
    )
    with col2:
        selected_ordering = st.selectbox(
            "Order molecules by:",
            ["Name", worst_first_option],
            key="structure_ordering_plot",
        )

    if selected_ordering == "Name":
        unique_structures = sorted(plot_molecule_df["Molecule"])
    else:
        unique_structures = list(
            plot_molecule_df.sort_values(
                selected_metric, ascending=sort_ascending, na_position="first"
            )["Molecule"]
        )

    if not unique_structures:
        st.markdown("**No molecules to display for this model**.")
        return

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

    spearman_corr = model_data_for_plot.spearman_correlation
    spearman_label = (
        f"Spearman ρ = {spearman_corr:.3f}, " if spearman_corr is not None else ""
    )

    # Create scatter plot
    scatter_chart = (
        alt.Chart(structure_df)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("Reference Energy:Q", title="Reference Energy (kcal/mol)"),
            y=alt.Y("Predicted Energy:Q", title="Predicted Energy (kcal/mol)"),
            tooltip=["Reference Energy:Q", "Predicted Energy:Q"],
        )
        .properties(
            width=600,
            height=400,
            title=(
                f"Model {selected_plot_model} - {selected_structure} "
                f"({spearman_label}{len(structure_df)} conformers)"
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

    st.altair_chart(final_chart, width="stretch")


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
