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

from mlipaudit.folding.folding import FoldingResult

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, FoldingResult]


def _data_to_dataframes(
    data: BenchmarkResultForMultipleModels,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_data = []
    agg_data = []

    for model_name, result in data.items():
        for molecule_result in result.molecules:
            for idx in range(len(molecule_result.rmsd_trajectory)):
                # Next line is just to stay inside line length maximum below
                prop_amino = molecule_result.proportion_folded_amino_acid[idx]

                plot_data.append({
                    "Model": model_name,
                    "Structure": molecule_result.structure_name,
                    "Frame": idx,
                    "RMSD": molecule_result.rmsd_trajectory[idx],
                    "TM score": molecule_result.tm_score_trajectory[idx],
                    "Rad. of Gyr.": molecule_result.radius_of_gyration[idx],
                    "Proportion of folded amino acids": prop_amino,
                    "DSSP match": molecule_result.match_secondary_structure[idx],
                })
                agg_data.append({
                    "Model": model_name,
                    "Structure": molecule_result.structure_name,
                    "min. RMSD": molecule_result.min_rmsd,
                    "max. TM score": molecule_result.max_tm_score,
                })

    df = pd.DataFrame(plot_data)
    df_agg = pd.DataFrame(agg_data)
    return df, df_agg


def _transform_dataframes_for_visualization(
    df: pd.DataFrame,
    df_agg: pd.DataFrame,
    selected_models: list[str],
    selected_structures: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_filtered = df[
        (df["Model"].isin(selected_models))
        & (df["Structure"].isin(selected_structures))
    ]
    df_agg_filtered = df_agg[
        (df_agg["Model"].isin(selected_models))
        & (df_agg["Structure"].isin(selected_structures))
    ]

    # Display model statistics table
    st.markdown("## Best model summary")

    # Calculate average metrics per model
    df_model_stats = (
        df_agg_filtered.groupby("Model")
        .agg({"min. RMSD": "mean", "max. TM score": "mean"})
        .round(4)
        .reset_index()
    )

    # Convert Model to string to ensure it's treated as categorical
    df_model_stats["Model"] = df_model_stats["Model"].astype(str)

    df_metrics = df_model_stats.set_index("Model")
    best_model_name = df_metrics["max. TM score"].idxmax()
    st.write(f"The best model is **{best_model_name}** based on maximum TM Score.")

    cols_metrics = st.columns(len(df_metrics.columns))
    for i, col in enumerate(df_metrics.columns):
        with cols_metrics[i]:
            st.metric(
                col,
                f"{df_metrics.loc[best_model_name, col]:.3f}",
            )

    # Box plots for average metrics across structures
    st.markdown("## Average metrics per model")

    # Ensure numeric values for aggregation
    df_agg_filtered_numeric = df_agg_filtered.copy()
    df_agg_filtered_numeric["min. RMSD"] = pd.to_numeric(
        df_agg_filtered_numeric["min. RMSD"], errors="coerce"
    )
    df_agg_filtered_numeric["max. TM score"] = pd.to_numeric(
        df_agg_filtered_numeric["max. TM score"], errors="coerce"
    )

    # Calculate averages across structures for each model
    avg_metrics = (
        df_agg_filtered_numeric.groupby("Model")
        .agg({"min. RMSD": "mean", "max. TM score": "mean"})
        .reset_index()
    )

    # Remove any rows with NaN values
    avg_metrics = avg_metrics.dropna()

    # Melt the data to create a long format for grouped bars
    metrics_long = avg_metrics.melt(
        id_vars=["Model"],
        value_vars=["min. RMSD", "max. TM score"],
        var_name="Metric",
        value_name="Value",
    )

    # Calculate rolling mean for proportion of folded amino acids
    df_filtered = df_filtered.copy()
    df_filtered["Proportion of folded amino acids"] = df_filtered.groupby([
        "Model",
        "Structure",
    ])[
        "Proportion of folded amino acids"
    ].transform(lambda x: x.rolling(window=21, center=True, min_periods=1).mean())

    # Calculate average trajectories across structures for each model
    avg_trajectories = (
        df_filtered.groupby(["Model", "Frame"])
        .agg({
            "RMSD": "mean",
            "TM score": "mean",
            "DSSP match": "mean",
            "Proportion of folded amino acids": "mean",
            "Rad. of Gyr.": "mean",
        })
        .reset_index()
    )

    # Calculate rolling mean for DSSP Match
    avg_trajectories["DSSP match smoothed"] = avg_trajectories.groupby("Model")[
        "DSSP match"
    ].transform(lambda x: x.rolling(window=21, center=True, min_periods=1).mean())

    return metrics_long, avg_trajectories


def folding_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for folding.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Protein folding trajectory analysis")
    st.sidebar.markdown("# Protein folding")

    st.markdown(
        "This module analyzes the folding trajectories of proteins in MLIP "
        "simulations. We track how RMSD, TM Score, DSSP evolve over time and Radius of "
        "Gyration change over time for three different structures: "
        "chignolin, trpcage and alanine coil (10-mer) "
        "simulations are started from an extended conformation."
    )

    st.markdown(
        "For more information, see the [docs](https://mlipaudit-dot-int-research"
        "-tpu.uc.r.appspot.com/benchmarks/biomolecules/folding/index.html)."
    )

    data = data_func()
    df, df_agg = _data_to_dataframes(data)

    unique_model_ids = list(data.keys())

    # Add "Select All" option
    all_models_option = st.sidebar.checkbox("Select all models", value=False)

    if all_models_option:
        model_select = unique_model_ids
    else:
        model_select = st.sidebar.multiselect(
            "Select models", unique_model_ids, default=unique_model_ids
        )

    selected_models = model_select if model_select else unique_model_ids

    unique_structures = list(set(df["Structure"].unique()))

    structure_select = st.sidebar.multiselect(
        "Select structures", unique_structures, default=unique_structures
    )
    selected_structures = structure_select if structure_select else unique_structures

    metrics_long, avg_trajectories = _transform_dataframes_for_visualization(
        df, df_agg, selected_models, selected_structures
    )

    # Create a grouped bar chart
    chart_grouped = (
        alt.Chart(metrics_long)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="Model", sort=None),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Metric:N", title="Metric"),
            xOffset=alt.XOffset("Metric:N"),
            tooltip=["Model:N", "Metric:N", "Value:Q"],
        )
        .properties(
            width=800,
            height=400,
        )
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart_grouped, use_container_width=True)
    buffer = io.BytesIO()
    chart_grouped.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="average_metrics_chart.png",
    )

    st.write("## Trajectory analysis over time")

    # 1. RMSD over time
    st.markdown("### RMSD over time ")
    st.markdown(
        "RMSD is computed between C-alpha atoms of the trajectory and the ground truth."
    )
    chart_rmsd = (
        alt.Chart(avg_trajectories)
        .mark_line(point=True)
        .encode(
            x=alt.X("Frame:Q", title="Frame"),
            y=alt.Y("RMSD:Q", title="RMSD (Å)"),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["Model", "Frame", "RMSD"],
        )
        .properties(
            width=800,
            height=400,
        )
    )
    st.altair_chart(chart_rmsd, use_container_width=True)
    buffer = io.BytesIO()
    chart_rmsd.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="rmsd_over_time.png",
    )

    # 2. TM Score over time
    st.markdown("### TM score over time vs ground truth")
    chart_tm = (
        alt.Chart(avg_trajectories)
        .mark_line(point=True)
        .encode(
            x=alt.X("Frame:Q", title="Frame"),
            y=alt.Y("TM_Score:Q", title="TM Score"),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["Model", "Frame", "TM score"],
        )
        .properties(
            width=800,
            height=400,
        )
    )
    st.altair_chart(chart_tm, use_container_width=True)
    buffer = io.BytesIO()
    chart_tm.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="tm_score_over_time.png",
    )

    # 3. DSSP Match over time
    st.markdown("### Secondary structure assigment match")

    st.markdown(
        "Secondary Structure assigment is compared to reference "
        "for each frame and the match is computed as the percentage "
        "of amino acids that are correctly assigned."
    )
    chart_secondary_structure = (
        alt.Chart(avg_trajectories)
        .mark_line(point=True)
        .encode(
            x=alt.X("Frame:Q", title="Frame"),
            y=alt.Y(
                "DSSP match smoothed:Q",
                title="DSSP match (smoothed)",
            ),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                "Model",
                "Frame",
                "DSSP match smoothed",
            ],
        )
        .properties(
            width=800,
            height=400,
        )
    )
    st.altair_chart(chart_secondary_structure, use_container_width=True)
    buffer = io.BytesIO()
    chart_secondary_structure.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="secondary_structure_assignment_match.png",
    )

    # 4. Secondary Structure Content over time (smoothed)
    st.markdown("### Proportion of folded amino acids over time")
    st.markdown(
        "Proportion of folded amino acids (C-alpha + C-beta) over time "
        "(smoothed over 20 frames for readability)."
    )
    chart_secondary_structure = (
        alt.Chart(avg_trajectories)
        .mark_line(point=True)
        .encode(
            x=alt.X("Frame:Q", title="Frame"),
            y=alt.Y(
                "Proportion of folded amino acids:Q",
                title="Proportion of folded amino acids (smoothed)",
            ),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                "Model",
                "Frame",
                "Proportion of folded amino acids",
            ],
        )
        .properties(
            width=800,
            height=400,
        )
    )
    st.altair_chart(chart_secondary_structure, use_container_width=True)
    buffer = io.BytesIO()
    chart_secondary_structure.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="prop_folded_amino_acids_over_time.png",
    )

    # 5. Radius of Gyration over time
    st.markdown("### Radius of gyration over time")
    chart_radius = (
        alt.Chart(avg_trajectories)
        .mark_line(point=True)
        .encode(
            x=alt.X("Frame:Q", title="Frame"),
            y=alt.Y("Radius:Q", title="Radius of Gyration (Å)"),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["Model", "Frame", "Rad. of Gyr."],
        )
        .properties(
            width=800,
            height=400,
        )
    )
    st.altair_chart(chart_radius, use_container_width=True)
    buffer = io.BytesIO()
    chart_radius.save(buffer, format="png", ppi=300)
    img_bytes = buffer.getvalue()
    st.download_button(
        label="Download plot",
        data=img_bytes,
        file_name="radius_of_gyration_over_time.png",
    )
