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
from typing import TypeAlias

import pandas as pd
import streamlit as st

from mlipaudit.benchmark import BenchmarkResult
from mlipaudit.io import OVERALL_SCORE_KEY_NAME

INTERNAL_MODELS_FILE_EXTENSION = "_int"
EXTERNAL_MODELS_FILE_EXTENSION = "_ext"
DEFAULT_IMAGE_DOWNLOAD_PPI = 300

ResultsOrScoresDict: TypeAlias = (
    dict[str, dict[str, float]] | dict[str, dict[str, BenchmarkResult]]
)
ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, BenchmarkResult]


def get_text_color(r: float, g: float, b: float) -> str:
    """Determine whether black or white text would be more readable.

    Args:
        r: Red channel (0-255).
        g: Green channel (0-255).
        b: Blue channel (0-255).

    Returns:
        "black" or "white" depending on which has better contrast.
    """
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "black"


def color_scores(val: int | float | str) -> str:
    """Applies a color gradient to scores. Scores closer
    to 1 (or max) will be darker blue, closer to 0 (or min) will be a
    lighter yellow.

    Will assume that scores are passed as integer, float, or string values.
    String values must either be convertable to float or are "N/A",
    in which case no formatting will be applied.

    Args:
        val: The cell value.

    Returns:
        The CSS style to apply to the cell.

    Raises:
        ValueError: If value is string, but neither 'N/A' nor convertible to float.
    """
    if val == "N/A":
        return ""  # No styling for cells that are "N/A"

    try:
        val = float(val)
    except ValueError:
        raise ValueError(
            "Scores that are given as string must either be "
            "convertible to float or 'N/A'."
        )

    # Normalize score from 0 to 1 for the gradient
    norm_val = max(0.0, min(1.0, val))  # Clamping values between 0 and 1

    # Background color: from light yellow to dark blue
    r_bg = int(255 - norm_val * (255 - 26))
    g_bg = int(255 - norm_val * (255 - 35))
    b_bg = int(224 - norm_val * (224 - 126))

    # Determine text color based on background luminance
    text_color = get_text_color(r_bg, g_bg, b_bg)

    return f"background-color: rgb({r_bg},{g_bg},{b_bg}); color: {text_color};"


def highlight_overall_score(s: pd.Series) -> list[str]:
    """Highlights the 'Overall score' column with a distinct, but
    colorblind-friendly background.

    Args:
        s: The Series representing the column.

    Returns:
        The list of styles to apply to each cell in the Series.
    """
    if s.name == OVERALL_SCORE_KEY_NAME.replace("_", " ").capitalize():
        # Specific background color for 'Overall score'
        bg_r, bg_g, bg_b = (173, 216, 230)  # RGB for Light Blue
        text_color = get_text_color(bg_r, bg_g, bg_b)
        return [
            f"background-color: rgb({bg_r},{bg_g},{bg_b}); color: {text_color};"
            for _ in s
        ]
    return ["" for _ in s]


def display_model_scores(df: pd.DataFrame) -> None:
    """Display model scores in a table. Expects either one of
    the columns to contain the model name or the index of the
    dataframe, both with the name 'Model name'.

    Raises:
        ValueError: If no column 'Score' or 'Model name'.
    """
    cols = df.columns.tolist()
    if "Score" in cols and cols[0] == "Model name":
        cols_index = 1
        hide_index = True
    elif "Score" in cols and "Model name" == df.axes[0].name:
        cols_index = 0
        hide_index = False
    else:
        raise ValueError("No 'Score' column found in DataFrame.")

    cols.remove("Score")
    cols.insert(cols_index, "Score")

    df_reordered = df[cols]

    st.dataframe(
        df_reordered.style.map(color_scores, subset=["Score"]).format(precision=3),
        hide_index=hide_index,
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
            scores_int[model_name] = model_scores
        elif model_name.endswith(EXTERNAL_MODELS_FILE_EXTENSION):
            scores_ext[model_name] = model_scores
    return scores_int, scores_ext


def remove_model_name_extensions_and_capitalize(
    results: dict[str, dict[str, BenchmarkResult]],
) -> dict[str, dict[str, BenchmarkResult]]:
    """Capitalize model names and remove extensions.

    Args:
        results: The scores' dictionary. The keys are the model names
            and the values dictionaries with the benchmark names
            as keys.

    Returns:
        The processed scores dictionary.
    """
    transformed_dict: dict[str, dict[str, BenchmarkResult]] = defaultdict(dict)
    for model_name, model_results in results.items():
        for benchmark_name, benchmark_result in model_results.items():
            transformed_dict[
                model_name.replace(INTERNAL_MODELS_FILE_EXTENSION, "")
                .replace(EXTERNAL_MODELS_FILE_EXTENSION, "")
                .replace("_", " ")
            ][benchmark_name] = benchmark_result

    return transformed_dict


def remove_model_name_extensions_and_capitalize_model_and_benchmark_names(
    results_or_scores: ResultsOrScoresDict,
) -> ResultsOrScoresDict:
    """Applies some transformations to a results or scoring dictionary
    for nicer printing.

    If the models have a trailing extension for the public leaderboard,
    this will be removed. For, benchmark names we will remove underscores and
    capitalize them.

    Args:
        results_or_scores: The scores or results dictionary. These have models as
                           keys and the value dictionaries have the benchmark names
                           as keys.

    Returns:
        The updated scores or results dictionary.
    """
    transformed_dict = defaultdict(dict)  # type: ignore
    for model_name, model_results_or_scores in results_or_scores.items():
        for (
            benchmark_name,
            benchmark_result_or_score,
        ) in model_results_or_scores.items():
            new_benchmark_name = benchmark_name.replace("_", " ").capitalize()
            transformed_dict[
                model_name.replace(INTERNAL_MODELS_FILE_EXTENSION, "")
                .replace(EXTERNAL_MODELS_FILE_EXTENSION, "")
                .replace("_", " ")
            ][new_benchmark_name] = benchmark_result_or_score

    return transformed_dict


def fetch_selected_models(available_models: list[str]) -> list[str]:
    """Fetch the intersection between the selected models and the
    available models for a given page.

    Args:
        available_models: List of available models for a specific benchmark.

    Returns:
        The list of selected models.
    """
    if st.session_state["max_selections"] == 1:
        selected_models = st.session_state["unique_model_names"]
    else:
        selected_models = st.session_state["selected_models"]

    return list(set(selected_models) & set(available_models))


def model_selection(unique_model_names: list[str]):
    """Handle the model selection across all pages. The selected
    models get saved to the session state. There is also an option
    'All' that will automatically select all the models. This should
    be called once on the app startup.

    Args:
        unique_model_names: The list of unique model names.
    """
    all_models = "All models"
    available_models = [all_models] + unique_model_names

    def options_select():
        if "selected_models" in st.session_state:
            if all_models in st.session_state["selected_models"]:
                # If 'All Models' is selected, force selection to only 'All Models'
                # and limit max_selections to 1
                st.session_state["selected_models"] = [all_models]
                st.session_state["max_selections"] = 1
            else:
                # If 'All Models' is not selected, allow all other models to be selected
                st.session_state["max_selections"] = len(available_models)

    if "unique_model_names" not in st.session_state:
        st.session_state["unique_model_names"] = unique_model_names
    if "all_model_names" not in st.session_state:
        st.session_state["all_model_names"] = available_models

    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = [all_models]
        st.session_state["max_selections"] = 1
    elif "max_selections" not in st.session_state:
        if all_models in st.session_state["selected_models"]:
            st.session_state["max_selections"] = 1
        else:
            st.session_state["max_selections"] = len(available_models)

    st.multiselect(
        label="Select Model(s)",
        options=available_models,
        key="selected_models",
        max_selections=st.session_state["max_selections"],
        on_change=options_select,
    )

    # failed_models = get_failed_models(data)
    # data = filter_failed_models(data)


def get_failed_models(data: BenchmarkResultForMultipleModels) -> list[str]:
    """Given a dictionary of benchmark results, fetch the list
     of failed models.

    Args:
        data: The dictionary of results for a given benchmark.

    Returns:
        A list of models that failed the benchmark.
    """
    return [model_name for model_name, result in data.items() if result.failed]


def display_failed_models(model_names: list[str]) -> None:
    """Write to the page the list of failed models, if any.

    Args:
        model_names: The list of model names.
    """
    markdown_list = "\n".join([f"* {model_name}" for model_name in model_names])
    if markdown_list:
        st.markdown("Models that failed to run: \n" + markdown_list)


def filter_failed_results(
    data: BenchmarkResultForMultipleModels,
) -> BenchmarkResultForMultipleModels:
    """Filter out failed models for a given benchmark.

    Args:
        data: The dictionary of results for a given benchmark.

    Returns:
        The filtered dictionary without the models that failed.
    """
    return {
        model_name: result for model_name, result in data.items() if not result.failed
    }
