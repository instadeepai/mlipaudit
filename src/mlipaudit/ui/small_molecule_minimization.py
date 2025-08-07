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

import pandas as pd
import streamlit as st

from mlipaudit.small_molecule_geometrics.small_molecule_minimization import (
    SmallMoleculeMinimizationBenchmark,
    SmallMoleculeMinimizationDatasetResult,
    SmallMoleculeMinimizationResult,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, SmallMoleculeMinimizationResult
]

EXPLODED_RMSD_THRESHOLD = 100.0
BAD_RMSD_THRESHOLD = 0.3


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
) -> pd.DataFrame:
    df_data = []
    for model_name, result in data.items():
        if model_name in selected_models:
            for dataset_prefix in SmallMoleculeMinimizationBenchmark.dataset_prefixes:
                model_dataset_result: SmallMoleculeMinimizationDatasetResult = getattr(
                    result, dataset_prefix
                )
                df_data.append({
                    "Model name": model_name,
                    "Dataset prefix": dataset_prefix,
                    "Average RMSD": model_dataset_result.avg_rmsd,
                    "Number of exploded structures": model_dataset_result.num_exploded,
                    "Number of bad RMSD scores": model_dataset_result.num_bad_rmsds,
                })

    return pd.DataFrame(df_data)


def small_molecule_minimization_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the small molecule minimization
    benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Small molecule energy minimization")
    st.sidebar.markdown("# Small molecule energy minimization")

    st.markdown(
        "Small molecule energy minimization benchmark. We run energy"
        " minimizations with "
        "our MLIP starting from reference structures extracted from the"
        " QM9 dataset and "
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
        "the QM9 dataset and the OpenFF dataset. To be able to verfify the MLIP's"
        "ability to represent charged systems, we split the two datasets into neutral "
        " and charged subsets. "
        "To ensure that the benchmark can be run within an acceptable time, we "
        "reduce the number of test structures to 100 for the neutral datasets"
        " and 10 for "
        "the charged datasets. The subsets are constructed so that the chemical "
        "diversity, "
        "as represented by Morgan fingerprints, is maximized. For each of these"
        " structures"
        ", an energy minimization is run."
    )

    st.markdown(
        "For more information, see the [docs](https://mlipaudit-dot-int-research-tpu.uc.r.appspot.com/benchmarks/small-molecules/small_molecule_rmsd.html)."
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

    df = _process_data_into_dataframe(data, selected_models)

    st.markdown("## Best model summary")

    best_model_row = df.loc[df["Average RMSD"].idxmin()]
    best_model_name = best_model_row["Model name"]

    st.markdown(f"The best model is **{best_model_name}** based on average RMSD.")
