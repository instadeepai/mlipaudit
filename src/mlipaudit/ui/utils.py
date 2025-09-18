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
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from mlipaudit.benchmark import BenchmarkResult

INTERNAL_MODELS_FILE_EXTENSION = "_int"
EXTERNAL_MODELS_FILE_EXTENSION = "_ext"
DEFAULT_IMAGE_DOWNLOAD_PPI = 300


def _color_score_blue_gradient(val):
    normalized_val = max(0, min(1, val))
    blue_intensity = int(normalized_val * 255)
    return f"background-color: rgb(255, 255, {blue_intensity})"


def display_model_scores(df: pd.DataFrame) -> None:
    """Display model scores in a table."""
    df_sorted = df[["Model name", "Score"]].sort_values(by="Score", ascending=False)
    st.dataframe(
        df_sorted.style.format(precision=3),
        hide_index=True,
    )


@st.cache_resource
def create_st_image(image_path: Path, caption: str | None = None) -> st.image:
    """Image creation helper that is cached.

    Args:
        image_path: Path to image.
        caption: Caption string. Can be None, which is the default.

    Returns:
        The streamlit image object.
    """
    return st.image(image_path, caption=caption)


def split_scores(
    scores: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Split the dictionary of scores into two, one for the
    internal models and the other for the external models.
    Also remove the extensions.

    Args:
        scores: The scores dictionary.

    Returns:
        The internal models and the external models.
    """
    scores_int, scores_ext = {}, {}
    for model_name, model_scores in scores.items():
        if model_name.endswith(INTERNAL_MODELS_FILE_EXTENSION):
            scores_int[model_name.replace(INTERNAL_MODELS_FILE_EXTENSION, "")] = (
                model_scores
            )
        elif model_name.endswith(EXTERNAL_MODELS_FILE_EXTENSION):
            scores_ext[model_name.replace(EXTERNAL_MODELS_FILE_EXTENSION, "")] = (
                model_scores
            )
    return scores_int, scores_ext


def update_model_and_benchmark_names(
    results_or_scores: dict[str, dict[str, float | BenchmarkResult]],
) -> dict[str, dict[str, float | BenchmarkResult]]:
    """Update the model and benchmark names
    for nicer printing. If the models have a trailing
    extension for the public leaderboard, this will be removed.
    Benchmark names will remove underscores and be
    capitalized.

    Args:
        results_or_scores: The scores or results dictionary.

    Returns:
        The updated scores or results dictionary.
    """
    new_data: dict[str, dict[str, float]] = defaultdict(dict)
    for model_name, model_results_or_scores in results_or_scores.items():
        for (
            benchmark_name,
            benchmark_result_or_score,
        ) in model_results_or_scores.items():
            new_benchmark_name = benchmark_name.replace("_", " ").capitalize()
            new_data[
                model_name.replace(INTERNAL_MODELS_FILE_EXTENSION, "").replace(
                    EXTERNAL_MODELS_FILE_EXTENSION, ""
                )
            ][new_benchmark_name] = benchmark_result_or_score

    return new_data
