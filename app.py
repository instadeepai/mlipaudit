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

import functools
import sys
from pathlib import Path

import streamlit as st

from mlipaudit.benchmark import Benchmark
from mlipaudit.conformer_selection import ConformerSelectionBenchmark
from mlipaudit.folding_stability import FoldingStabilityBenchmark
from mlipaudit.io import load_benchmark_results_from_disk
from mlipaudit.tautomers import TautomersBenchmark
from mlipaudit.ui import (
    conformer_selection_page,
    folding_stability_page,
    tautomers_page,
)

BENCHMARKS: list[type[Benchmark]] = [
    ConformerSelectionBenchmark,
    TautomersBenchmark,
    FoldingStabilityBenchmark,
]


def _data_func_from_key(key, results_data):
    def _func():
        return {model: benchmarks[key] for model, benchmarks in results_data.items()}

    return _func


if len(sys.argv) < 2:
    raise RuntimeError(
        "You must provide the results directory as a command line argument, "
        "like this: streamlit run app.py /path/to/results"
    )

if not Path(sys.argv[1]).exists():
    raise RuntimeError("The specified results directory does not exist.")

data = load_benchmark_results_from_disk(sys.argv[1], BENCHMARKS)

small_molecule_conformers = st.Page(
    functools.partial(
        conformer_selection_page,
        data_func=_data_func_from_key("conformer_selection", data),
    ),
    title="Small molecule conformers",
    url_path="conformer_selection",
)

tautomers = st.Page(
    functools.partial(
        tautomers_page,
        data_func=_data_func_from_key("tautomers", data),
    ),
    title="Tautomers",
    url_path="tautomers",
)

folding_stability = st.Page(
    functools.partial(
        folding_stability_page,
        data_func=_data_func_from_key("folding_stability", data),
    ),
    title="Protein folding stability",
    url_path="protein_folding_stability",
)

# Define page categories
page_categories = {
    "Small Molecules": [
        small_molecule_conformers,
        tautomers,
    ],
    "Biomolecules": [
        folding_stability,
    ],
}

# Create sidebar container for category selection
with st.sidebar.container():
    st.markdown("### Select Analysis Category")
    selected_category = st.selectbox(
        "Choose a category:",
        ["All Categories"] + list(page_categories.keys()),
        help="Filter pages by category",
    )

# Filter pages based on selection
if selected_category == "All Categories":
    pages_to_show = page_categories["Small Molecules"] + page_categories["Biomolecules"]

else:
    pages_to_show = page_categories[selected_category]

# Set up navigation in main area
pg = st.navigation(pages_to_show)

# Run the selected page
pg.run()
