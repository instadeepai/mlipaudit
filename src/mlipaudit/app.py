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
from typing import Callable

import streamlit as st
from streamlit import runtime as st_runtime
from streamlit.web import cli as st_cli

from mlipaudit.benchmark import Benchmark, BenchmarkResult
from mlipaudit.bond_length_distribution import BondLengthDistributionBenchmark
from mlipaudit.conformer_selection import ConformerSelectionBenchmark
from mlipaudit.dihedral_scan import DihedralScanBenchmark
from mlipaudit.folding_stability import FoldingStabilityBenchmark
from mlipaudit.io import load_benchmark_results_from_disk, load_scores_from_disk
from mlipaudit.noncovalent_interactions import NoncovalentInteractionsBenchmark
from mlipaudit.reactivity import ReactivityBenchmark
from mlipaudit.ring_planarity import RingPlanarityBenchmark
from mlipaudit.sampling import SamplingBenchmark
from mlipaudit.scaling import ScalingBenchmark
from mlipaudit.small_molecule_minimization import (
    SmallMoleculeMinimizationBenchmark,
)
from mlipaudit.solvent_radial_distribution import SolventRadialDistributionBenchmark
from mlipaudit.stability import StabilityBenchmark
from mlipaudit.tautomers import TautomersBenchmark
from mlipaudit.ui import (
    bond_length_distribution_page,
    conformer_selection_page,
    dihedral_scan_page,
    folding_stability_page,
    leaderboard_page,
    noncovalent_interactions_page,
    reactivity_page,
    ring_planarity_page,
    sampling_page,
    scaling_page,
    small_molecule_minimization_page,
    solvent_radial_distribution_page,
    stability_page,
    tautomers_page,
    water_radial_distribution_page,
)
from mlipaudit.water_radial_distribution import (
    WaterRadialDistributionBenchmark,
)

BENCHMARKS: list[type[Benchmark]] = [
    ConformerSelectionBenchmark,
    DihedralScanBenchmark,
    NoncovalentInteractionsBenchmark,
    TautomersBenchmark,
    RingPlanarityBenchmark,
    SmallMoleculeMinimizationBenchmark,
    FoldingStabilityBenchmark,
    BondLengthDistributionBenchmark,
    SamplingBenchmark,
    WaterRadialDistributionBenchmark,
    SolventRadialDistributionBenchmark,
    StabilityBenchmark,
    ReactivityBenchmark,
    ScalingBenchmark,
]


def _data_func_from_key(
    benchmark_name: str, results_data: dict[str, dict[str, BenchmarkResult]]
) -> Callable[[], dict[str, BenchmarkResult]]:
    """Return a function that when called filters `results_data` and
    returns a dictionary where the keys correspond to the model names
    and the values the result of the benchmark given by ` benchmark_name`.
    """

    def _func():
        results = {}
        for model, benchmarks in results_data.items():
            if benchmarks.get(benchmark_name) is not None:
                results[model] = benchmarks.get(benchmark_name)
        return results

    return _func


def main():
    """Main of our UI app.

    Raises:
        RuntimeError: if results directory is not passed as argument.
    """
    if len(sys.argv) < 2:
        raise RuntimeError(
            "You must provide the results directory as a command line argument, "
            "like this: mlipauditapp /path/to/results"
        )
    remote = False
    if sys.argv[1] == "__hf":
        remote = True
        results_dir = sys.argv[2]
    else:
        if not Path(sys.argv[1]).exists():
            raise RuntimeError("The specified results directory does not exist.")
        results_dir = sys.argv[1]

    data = load_benchmark_results_from_disk(results_dir, BENCHMARKS)
    scores = load_scores_from_disk(scores_dir=results_dir)

    leaderboard = st.Page(
        functools.partial(leaderboard_page, scores=scores, remote=remote),
        title="Leaderboard",
        icon=":material/trophy:",
        default=True,
    )

    conformer_selection = st.Page(
        functools.partial(
            conformer_selection_page,
            data_func=_data_func_from_key("conformer_selection", data),
        ),
        title="Conformer selection",
        url_path="conformer_selection",
    )
    dihedral_scan = st.Page(
        functools.partial(
            dihedral_scan_page,
            data_func=_data_func_from_key("dihedral_scan", data),
        ),
        title="Dihedral scan",
        url_path="dihedral_scan",
    )

    tautomers = st.Page(
        functools.partial(
            tautomers_page,
            data_func=_data_func_from_key("tautomers", data),
        ),
        title="Tautomers",
        url_path="tautomers",
    )
    noncovalent_interactions = st.Page(
        functools.partial(
            noncovalent_interactions_page,
            data_func=_data_func_from_key("noncovalent_interactions", data),
        ),
        title="Noncovalent Interactions",
        url_path="noncovalent_interactions",
    )
    ring_planarity = st.Page(
        functools.partial(
            ring_planarity_page,
            data_func=_data_func_from_key("ring_planarity", data),
        ),
        title="Ring planarity",
        url_path="ring_planarity",
    )

    small_molecule_minimization = st.Page(
        functools.partial(
            small_molecule_minimization_page,
            data_func=_data_func_from_key("small_molecule_minimization", data),
        ),
        title="Small molecule minimization",
        url_path="small_molecule_minimization",
    )

    reactivity = st.Page(
        functools.partial(
            reactivity_page,
            data_func=_data_func_from_key("reactivity", data),
        ),
        title="Reactivity",
        url_path="reactivity",
    )

    folding_stability = st.Page(
        functools.partial(
            folding_stability_page,
            data_func=_data_func_from_key("folding_stability", data),
        ),
        title="Protein folding stability",
        url_path="protein_folding_stability",
    )

    bond_length_distribution = st.Page(
        functools.partial(
            bond_length_distribution_page,
            data_func=_data_func_from_key("bond_length_distribution", data),
        ),
        title="Bond length distribution",
        url_path="bond_length_distribution",
    )

    sampling = st.Page(
        functools.partial(
            sampling_page,
            data_func=_data_func_from_key("sampling", data),
        ),
        title="Protein sampling",
        url_path="sampling",
    )

    water_radial_distribution = st.Page(
        functools.partial(
            water_radial_distribution_page,
            data_func=_data_func_from_key("water_radial_distribution", data),
        ),
        title="Water radial distribution function",
        url_path="water_radial_distribution_function",
    )

    solvent_radial_distribution = st.Page(
        functools.partial(
            solvent_radial_distribution_page,
            data_func=_data_func_from_key("solvent_radial_distribution", data),
        ),
        title="Solvent radial distribution",
        url_path="solvent_radial_distribution",
    )

    stability = st.Page(
        functools.partial(
            stability_page,
            data_func=_data_func_from_key("stability", data),
        ),
        title="Stability",
        url_path="stability",
    )

    scaling = st.Page(
        functools.partial(
            scaling_page,
            data_func=_data_func_from_key("scaling", data),
        ),
        title="Scaling",
        url_path="scaling",
    )

    # Define page categories
    page_categories = {
        "Small Molecules": [
            conformer_selection,
            dihedral_scan,
            tautomers,
            noncovalent_interactions,
            ring_planarity,
            small_molecule_minimization,
            bond_length_distribution,
            water_radial_distribution,
            solvent_radial_distribution,
            reactivity,
        ],
        "Biomolecules": [
            folding_stability,
            sampling,
        ],
        "General": [stability, scaling],
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
        pages_to_show = [leaderboard] + (
            page_categories["Small Molecules"]
            + page_categories["Biomolecules"]
            + page_categories["General"]
        )

    else:
        pages_to_show = [leaderboard] + page_categories[selected_category]

    # Set up navigation in main area
    pg = st.navigation(pages_to_show)

    # Run the selected page
    pg.run()


def launch_app():
    """Figures out whether run by streamlit or not. Then calls `main()`."""
    if st_runtime.exists():
        main()
    else:
        original_args_without_exec = sys.argv[1:]
        sys.argv = ["streamlit", "run", __file__] + original_args_without_exec
        sys.exit(st_cli.main())


if __name__ == "__main__":
    launch_app()
