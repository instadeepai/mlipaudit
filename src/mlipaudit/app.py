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
from streamlit.web import cli as st_cli

from mlipaudit.benchmark import BenchmarkResult
from mlipaudit.benchmarks import (
    BENCHMARK_CATEGORIES,
    BENCHMARKS,
    BENCHMARKS_TO_SKIP_FOR_PUBLIC_LEADERBOARD,
)
from mlipaudit.io import load_benchmark_results_from_disk, load_scores_from_disk
from mlipaudit.ui import leaderboard_page
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    model_selection,
    remove_model_name_extensions_and_capitalize,
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


def _parse_app_args(argvs: list[str]) -> tuple[str, bool]:
    """Parse the command line arguments for the app.

    Args:
        argvs: The command line arguments.

    Returns:
        The parsed arguments.

    Raises:
        RuntimeError: if results directory is not passed as argument.
    """
    if len(argvs) < 2:
        raise RuntimeError(
            "You must provide the results directory as a command line argument, "
            "like this: mlipaudit gui /path/to/results"
        )
    is_public = False
    if len(argvs) == 3 and argvs[2] == "__public":
        is_public = True

    if not Path(argvs[1]).exists():
        raise RuntimeError("The specified results directory does not exist.")

    results_dir = argvs[1]
    return results_dir, is_public


def main() -> None:
    """Main of our UI app.

    Raises:
        RuntimeError: if results directory is not passed as argument.
    """
    results_dir, is_public = _parse_app_args(argvs=sys.argv)

    results = load_benchmark_results_from_disk(results_dir, BENCHMARKS)
    scores = load_scores_from_disk(scores_dir=results_dir)

    # Some benchmarks are still in beta and are not displayed in the public leaderboard
    if is_public:
        for _, model_scores in scores.items():
            for benchmark in BENCHMARKS_TO_SKIP_FOR_PUBLIC_LEADERBOARD:
                model_scores.pop(benchmark.name, None)

    leaderboard = st.Page(
        functools.partial(leaderboard_page, scores=scores, is_public=is_public),
        title="Leaderboard",
        icon=":material/trophy:",
        default=True,
    )

    # For the remaining pages, update the model names
    results = remove_model_name_extensions_and_capitalize(results)

    benchmark_pages = {}
    for page_wrapper in UIPageWrapper.__subclasses__():
        name = page_wrapper.get_benchmark_class().name
        if is_public and name in {
            b.name for b in BENCHMARKS_TO_SKIP_FOR_PUBLIC_LEADERBOARD
        }:
            continue
        benchmark_pages[name] = st.Page(
            functools.partial(
                page_wrapper.get_page_func(),
                data_func=_data_func_from_key(name, results),  # type: ignore
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

    # Add model selection
    with st.sidebar:
        model_selection(unique_model_names=list(results.keys()))

    # Set up navigation in main area
    pg = st.navigation(pages_to_show)

    # Run the selected page
    pg.run()


def launch_app(results_dir: str, is_public: bool) -> None:
    """Figures out whether run by streamlit or not. Then calls `main()`."""
    args = [results_dir]
    if is_public:
        args.append("__public")
    sys.argv = ["streamlit", "run", __file__] + args
    sys.exit(st_cli.main())


if __name__ == "__main__":
    main()
