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

from mlipaudit.benchmark import BenchmarkResult
from mlipaudit.benchmarks import BENCHMARK_CATEGORIES, BENCHMARKS
from mlipaudit.io import load_benchmark_results_from_disk, load_scores_from_disk
from mlipaudit.ui import leaderboard_page
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    remove_model_name_extensions_and_capitalize_model_and_benchmark_names,
)


def _data_func_from_key(
    benchmark_name: str, results_data: dict[str, dict[str, BenchmarkResult]]
) -> Callable[[], dict[str, BenchmarkResult]]:
    """Return a function that when called filters `results_data` and
    returns a dictionary where the keys correspond to the model names
    and the values the result of the benchmark given by `benchmark_name`.
    """

    def _func():
        results = {}
        for model, benchmarks in results_data.items():
            if benchmarks.get(benchmark_name) is not None:
                results[model] = benchmarks.get(benchmark_name)
        return results

    return _func


def _get_pages_for_category(
    category: str, benchmark_pages: dict[str, st.Page]
) -> list[st.Page]:
    """Fetches all the benchmark pages for a specific category from a
    dictionary of all benchmark pages.

    Args:
        category: Benchmark category.
        benchmark_pages: A dictionary of streamlit pages. Keys are benchmark names.

    Returns:
        The pages for a given category as a list.
    """
    return [
        page
        for name, page in benchmark_pages.items()
        if name in [b.name for b in BENCHMARK_CATEGORIES[category]]
    ]


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
    is_public = False
    if len(sys.argv) == 3 and sys.argv[2] == "__hf":
        is_public = True
    else:
        if not Path(sys.argv[1]).exists():
            raise RuntimeError("The specified results directory does not exist.")

    results_dir = sys.argv[1]

    results = load_benchmark_results_from_disk(results_dir, BENCHMARKS)
    scores = load_scores_from_disk(scores_dir=results_dir)

    if is_public:
        remove_model_name_extensions_and_capitalize_model_and_benchmark_names(results)

    leaderboard = st.Page(
        functools.partial(leaderboard_page, scores=scores, is_public=is_public),
        title="Leaderboard",
        icon=":material/trophy:",
        default=True,
    )

    benchmark_pages = {}
    for page_wrapper in UIPageWrapper.__subclasses__():
        name = page_wrapper.get_benchmark_class().name
        benchmark_pages[name] = st.Page(
            functools.partial(
                page_wrapper.get_page_func(),
                data_func=_data_func_from_key(name, results),
            ),
            title=name.replace("_", " ").capitalize(),
            url_path=name,
        )

    # Define page categories
    categories_in_order = [
        "Small Molecules",
        "Biomolecules",
        "Molecular Liquids",
        "General",
    ]
    # Add other (possibly new) categories in any order after that
    categories_in_order += [
        cat for cat in BENCHMARK_CATEGORIES if cat not in categories_in_order
    ]
    page_categories = {
        category: _get_pages_for_category(category, benchmark_pages)
        for category in categories_in_order
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
        pages_to_show = [leaderboard]
        for category in categories_in_order:
            pages_to_show += page_categories[category]
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
