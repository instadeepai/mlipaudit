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

from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    BondLengthDistributionResult,
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
    ModelName, BondLengthDistributionResult
]


SHORT_LABELS = {
    "carbon-carbon (single)": "C-C",
    "carbon-carbon (double)": "C=C",
    "carbon-carbon (triple)": "C#C",
    "carbon-oxygen (single)": "C-O",
    "carbon-oxygen (double)": "C=O",
    "carbon-nitrogen (single)": "C-N",
    "carbon-nitrogen (triple)": "C#N",
    "carbon-fluorine (single)": "C-F",
}


def bond_length_distribution_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for bond length distribution.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Bond length distribution")

    st.markdown(
        "The benchmark runs short simulations of small molecules to check whether the "
        "correct bond lengths of typical bonds found in organic small molecules are "
        "preserved. This is an important test to see if the MLIP respects basic "
        "chemistry throughout simulations. For every bond type, the benchmark runs"
        " a short simulation of a test molecule and the bond length of that bond"
        " type is recorded. The key metric is the average deviation of the bond"
        " length throughout the simulation. This value should not exceed 0.025 Å."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit/benchmarks/"
        "small_molecules/bond_length_distribution.html)."
    )

    # Download data and get model names
    if "bond_length_distribution_cached_data" not in st.session_state:
        st.session_state.bond_length_distribution_cached_data = data_func()

    # Retrieve the data from the session state
    data: BenchmarkResultForMultipleModels = (
        st.session_state.bond_length_distribution_cached_data
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

    distribution_data = [
        {
            "Model name": model_name,
            "Average deviation (Å)": result.avg_deviation,
            "Score": result.score,
        }
        for model_name, result in data.items()
        if model_name in selected_models
    ]

    st.markdown("## Summary statistics")

    df = pd.DataFrame(distribution_data)

    df.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df)

    st.markdown("## Bond length deviation distribution per model")

    # Get all unique ring types from the data
    all_bond_types_set: set[str] = set()

    for model_name, result in data.items():
        all_bond_types_set.update(mol.molecule_name for mol in result.molecules)
    all_bond_types = sorted(list(all_bond_types_set))

    # Bond type selection dropdown
    selected_bond_type = st.selectbox(
        "Select bond type:", all_bond_types, index=0 if all_bond_types else None
    )

    if selected_bond_type:
        # Transform the data for the selected ring type
        plot_data = []

        for model_name, result in data.items():
            if model_name in selected_models:
                for mol in result.molecules:
                    if selected_bond_type == mol.molecule_name and not mol.failed:
                        for bond_length in mol.deviation_trajectory:  # type: ignore
                            plot_data.append({
                                "Model name": model_name,
                                "Bond length": bond_length,
                            })
        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            # Create the boxplot chart
            chart = (
                alt.Chart(df_plot)
                .mark_boxplot(extent="min-max", size=50, color="darkgrey")
                .encode(
                    x=alt.X(
                        "Model name:N",
                        title="Model",
                        axis=alt.Axis(labelAngle=-45, labelLimit=100),
                    ),
                    y=alt.Y(
                        "Bond length:Q",
                        title="Bond length deviation (Å)",
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color(
                        "Model name:N",
                        title="Model",
                        legend=alt.Legend(orient="top"),
                    ),
                )
                .properties(
                    width=600,
                    height=400,
                )
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Please select a bond type to view the distribution.")


class BondLengthDistributionPageWrapper(UIPageWrapper):
    """Page wrapper around bond length distribution benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return bond_length_distribution_page

    @classmethod
    def get_benchmark_class(cls) -> type[BondLengthDistributionBenchmark]:  # noqa: D102
        return BondLengthDistributionBenchmark
