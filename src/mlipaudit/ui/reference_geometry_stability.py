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

from typing import Callable, TypeAlias

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.benchmarks.reference_geometry_stability.reference_geometry_stability import (  # noqa: E501
    DATASET_PREFIXES,
    ReferenceGeometryStabilityBenchmark,
    ReferenceGeometryStabilityDatasetResult,
    ReferenceGeometryStabilityResult,
)
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, ReferenceGeometryStabilityResult
]

EXPLODED_RMSD_THRESHOLD = 100.0
BAD_RMSD_THRESHOLD = 0.3

DATASET_NAME_MAP = {
    "openff_neutral": "Openff neutral",
    "openff_charged": "Openff charged",
}


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    df_data = []
    for model_name, result in data.items():
        if model_name in selected_models:
            for dataset_prefix in DATASET_PREFIXES:
                model_dataset_result: ReferenceGeometryStabilityDatasetResult = getattr(
                    result, dataset_prefix
                )
                if model_dataset_result.failed:
                    continue
                df_data.append({
                    "Model name": model_name,
                    "Score": result.score,
                    "Dataset": DATASET_NAME_MAP[dataset_prefix],
                    "Average RMSD (Å)": model_dataset_result.avg_rmsd,
                    "Number of exploded structures": model_dataset_result.num_exploded,
                    "Number of bad RMSD scores": model_dataset_result.num_bad_rmsds,
                })

    return pd.DataFrame(df_data)


def reference_geometry_stability_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the reference geometry stability
    benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Reference geometry stability")

    st.markdown(
        "Reference geometry stability benchmark. We run energy"
        " minimizations with "
        "our MLIP starting from reference structures extracted from the"
        " OpenFF dataset and "
        "calculate after the minimization, how much the atomic positions"
        " of the "
        "heavy atoms deviate from the reference structure. The key metric"
        " for measuring "
        "the deviation is the RMSD. This benchmark assesses if the MLIP"
        " is able to "
        "retain the QM reference structure's geometry."
    )

    st.markdown(
        "Here, we test this ability on two datasets of organic small molecules: "
        "the  OpenFF dataset. To be able to verfify the MLIP's "
        "ability to represent charged systems, we split the two datasets into neutral "
        " and charged subsets. "
        "To ensure that the benchmark can be run within an acceptable time, we "
        "reduce the number of test structures to 200 for the neutral datasets"
        " and 20 for "
        "the charged datasets. The subsets are constructed so that the chemical "
        "diversity, "
        "as represented by Morgan fingerprints, is maximized. For each of these"
        " structures"
        ", an energy minimization is run."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit/benchmarks/"
        "small_molecules/reference_geometry_stability.html)."
    )

    # Download data and get model names
    if "reference_geometry_stability_cached_data" not in st.session_state:
        st.session_state.reference_geometry_stability_cached_data = data_func()

    # Retrieve the data from the session state
    data = st.session_state.reference_geometry_stability_cached_data

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
    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## Average RMSD per model and dataset")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Dataset:N",
                title="Dataset",
                axis=alt.Axis(labelAngle=-45, labelLimit=100),
            ),
            y=alt.Y("Average RMSD (Å):Q", title="Average RMSD (Å)"),
            color=alt.Color("Model name:N", title="Model"),
            xOffset=alt.XOffset("Model name:N"),
            tooltip=[
                alt.Tooltip("Dataset:N"),
                alt.Tooltip("Model name:N", title="Model"),
                alt.Tooltip(
                    "Average RMSD (Å):Q", title="Average RMSD (Å)", format=".3f"
                ),
                alt.Tooltip(
                    "Number of exploded structures:Q", title="Exploded structures"
                ),
                alt.Tooltip("Number of bad RMSD scores:Q", title="Bad RMSD structures"),
            ],
        )
        .properties(width=600, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("## Exploded structures report")
    st.markdown(
        "If any of the energy minimizations exploded, "
        "we list here the number of exploded structures per model and dataset."
    )

    # Create exploded structures table
    df_exploded = (
        df.pivot(
            index="Dataset",
            columns="Model name",
            values="Number of exploded structures",
        )
        .fillna(0)
        .astype(int)
    )

    # Check if all entries are zero
    if df_exploded.values.sum() == 0:
        st.markdown(
            "**No exploded structures found:** "
            "All structures remained stable during minimization."
        )
    else:
        st.dataframe(df_exploded, use_container_width=True)

    st.markdown("## Bad RMSD report")
    st.markdown(
        "If any of the structures after energy minimization have RMSD > 0.3 Å, "
        "we list here the number of such structures per model and dataset."
    )

    df_bad_rmsd = (
        df.pivot(
            index="Dataset", columns="Model name", values="Number of bad RMSD scores"
        )
        .fillna(0)
        .astype(int)
    )

    if df_bad_rmsd.values.sum() == 0:
        st.markdown(
            "**No structures with RMSD > 0.3 Å found:** "
            "All structures converged with good RMSD."
        )
    else:
        st.dataframe(df_bad_rmsd, use_container_width=True)


class ReferenceGeometryStabilityWrapper(UIPageWrapper):
    """Page wrapper for reference geometry stability benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return reference_geometry_stability_page

    @classmethod
    def get_benchmark_class(cls) -> type[ReferenceGeometryStabilityBenchmark]:  # noqa: D102
        return ReferenceGeometryStabilityBenchmark
