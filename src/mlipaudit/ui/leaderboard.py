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

import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import (
    BENCHMARK_CATEGORIES,
    BENCHMARK_WITH_SCORES_CATEGORIES,
    BENCHMARK_WITH_SCORES_CATEGORIES_PUBLIC,
)
from mlipaudit.io import OVERALL_SCORE_KEY_NAME
from mlipaudit.ui.utils import (
    color_scores,
    highlight_overall_score,
    remove_model_name_extensions_and_capitalize_model_and_benchmark_names,
    split_scores,
)


@st.cache_data
def parse_scores_dict_into_df(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Parse the scores into a dataframe, using model name as index.

    Args:
        scores: The parsed scores dictionary.

    Returns:
        A dataframe with model name as index, and columns ordered as Overall
            score, with the remaining cols.
    """
    df_data = {}  # Use dict to store index -> row data
    for model_name, benchmark_scores in scores.items():
        row: dict[str, float] = {}
        for benchmark_name, score_value in benchmark_scores.items():
            if score_value is not None:
                row[benchmark_name] = score_value
        df_data[model_name] = row

    # Create DataFrame with model_name as index
    df = pd.DataFrame.from_dict(df_data, orient="index")
    df.index.name = "Model name"  # Set a name for the index

    # Reorder columns: Overall score first
    cols = df.columns.tolist()
    if "Overall score" in cols:
        cols.remove("Overall score")
        cols = ["Overall score"] + cols

    df = df[cols]

    return df


def _color_overall_score(val):
    color = (
        "background-color: #6D9EEB"
        if val > 0.7
        else "background-color: #9FC5E8"
        if val > 0.3
        else "background-color: #C9DAF8"
    )
    return color


def _color_individual_score(val):
    color = (
        "background-color: lightgreen"
        if val > 0.7
        else "background-color: yellow"
        if val > 0.3
        else "background-color: lightcoral"
    )
    return color


def _group_score_df_by_benchmark_category(
    score_df: pd.DataFrame, is_public: bool
) -> pd.DataFrame:
    for category in BENCHMARK_CATEGORIES:
        names = [
            b.name.replace("_", " ").capitalize()
            for b in BENCHMARK_CATEGORIES[category]  # type: ignore
        ]
        names_filtered = [b for b in names if b in score_df.columns]

        num_scores_category = BENCHMARK_WITH_SCORES_CATEGORIES[category]
        if is_public:
            num_scores_category = BENCHMARK_WITH_SCORES_CATEGORIES_PUBLIC[category]

        score_df[category] = score_df[names_filtered].sum(axis=1) / num_scores_category
        score_df = score_df.drop(columns=names_filtered)

    columns_in_order = [
        "Overall score",
        "Small Molecules",
        "Biomolecules",
        "Molecular Liquids",
        "General",
    ]
    if is_public:
        columns_in_order.insert(1, "Model Type")

    # Add other (possibly new) categories in any order after that
    columns_in_order += [
        cat for cat in BENCHMARK_CATEGORIES if cat not in columns_in_order
    ]

    return score_df[columns_in_order]


def leaderboard_page(
    scores: dict[str, dict[str, float]],
    is_public: bool = False,
) -> None:
    """Leaderboard page. Takes the preprocessed scores and displays them.
    If `is_public` is False, display all the results in a single table,
    otherwise for the remote leaderboard, separate the scores into two tables.

    Args:
        scores: The preprocessed scores. The first keys are the model names
            and the second keys are the benchmark names.
        is_public: Whether displaying locally or for the public leaderboard.
    """
    st.markdown("# MLIPAudit")

    st.markdown(
        """
        MLIPAudit is a Python tool for benchmarking and validating
        Machine Learning Interatomic Potentials (MLIP) models,
        specifically those based on the [mlip](https://github.com/instadeepai/mlip)
        library. It aims to cover a wide range of use cases and difficulties, providing
        users with a comprehensive overview of the performance of their models.
        """
    )

    st.markdown("## How to Interpret MLIPAudit Scores")

    st.markdown(
        """
        The **Model Scores** section provides several aggregated metrics to help you
        quickly understand a model's overall performance:
        - **Overall** – Average across all benchmarks
        - **Small Molecules** – Average performance on small-molecule benchmarks
        - **Biomolecules** – Average performance on biomolecular benchmarks
        - **Molecular** Liquids – Average performance on liquid-phase benchmarks
        - **General** – Average performance on general stability and dynamics benchmarks

        **Important:**
        Aggregated scores offer a convenient overview, but they should not be
        interpreted in isolation. If a model cannot run a benchmark — typically due
        to missing chemical element coverage — it receives a score of zero for that
        task. This affects its aggregated category and overall scores.
        This penalty does **not** reflect the intrinsic quality of the model; instead,
        it highlights limitations in **chemical generality** or **domain coverage**.
        Furthermore, the overall score is influenced more heavily by the small molecule
        benchmarks as there are more of them.

        For any real application, we recommend reviewing the
        **individual benchmark results** to evaluate whether a model is suitable
        for your specific downstream task.

        It is also worth noting that we cannot guarantee that there is no overlap
        between training set and our benchmark test systems for models trained on
        the OMol25 dataset (e.g., UMA-small).

        You can find a detailed explanation of how scores are computed in the
        **MLIPAudit code documentation**
        [here](https://instadeepai.github.io/mlipaudit/scores).
        """
    )

    # 1. Conditional Data Preprocessing and DataFrame Creation
    if is_public:
        # PUBLIC LEADERBOARD: Split, preprocess, add 'Model Type', and combine
        scores_int, scores_ext = split_scores(scores)

        scores_int = (
            remove_model_name_extensions_and_capitalize_model_and_benchmark_names(  # type: ignore
                scores_int
            )
        )
        scores_ext = (
            remove_model_name_extensions_and_capitalize_model_and_benchmark_names(  # type: ignore
                scores_ext
            )
        )

        df_int = parse_scores_dict_into_df(scores_int)
        df_ext = parse_scores_dict_into_df(scores_ext)

        df_int["Model Type"] = "InstaDeep"
        df_ext["Model Type"] = "Community"

        df_main = pd.concat([df_int, df_ext], ignore_index=False)
    else:
        # LOCAL LEADERBOARD: Preprocess all, create single DataFrame
        scores = remove_model_name_extensions_and_capitalize_model_and_benchmark_names(
            scores
        )  # type: ignore
        df_main = parse_scores_dict_into_df(scores)

    # 2. Main Table Display (Common Logic)
    df_main.sort_values(
        by=OVERALL_SCORE_KEY_NAME.replace("_", " ").capitalize(),
        ascending=False,
        inplace=True,
    )

    df_grouped_main = _group_score_df_by_benchmark_category(df_main, is_public)

    st.markdown("## Model Scores")

    # Columns to apply coloring to in the grouped table
    color_subset_cols = [
        "Overall score",
        "Small Molecules",
        "Biomolecules",
        "Molecular Liquids",
        "General",
    ]

    # Ensure only existing columns are passed to subset
    valid_color_subset = [
        col for col in color_subset_cols if col in df_grouped_main.columns
    ]

    df_grouped_main = df_grouped_main.apply(
        lambda col: col.map(
            lambda val: "N/A"
            if pd.isna(val)
            # Make sure that string columns are not converted
            else (val if type(val) is str else f"{val:.2f}")
        )
    )

    styled_df = df_grouped_main.style.map(
        color_scores,
        subset=pd.IndexSlice[
            :,
            valid_color_subset,
        ],
    ).apply(highlight_overall_score, axis=0)

    st.dataframe(styled_df, hide_index=False)

    st.markdown(
        """
        <small>
            <b> Color Scheme Note: </b> Scores are colored on a gradient from
            light yellow (lower scores) to dark blue (higher scores). The 'Overall
            score' column is additionally highlighted with a light blue background
            for emphasis. This scheme is chosen for its general
            colorblind-friendliness.
        </small>
    """,
        unsafe_allow_html=True,
    )

    # 3. Individual Benchmark Tables Display (Common Logic with Conditional Columns)

    for category in BENCHMARK_CATEGORIES:
        st.markdown(f"### {category} Benchmarks")

        # Determine benchmark names in the DataFrame for this category
        names = [
            b.name.replace("_", " ").capitalize()
            for b in BENCHMARK_CATEGORIES[category]  # type: ignore
        ]
        names_filtered = [b for b in names if b in df_main.columns]

        if not names_filtered:
            st.markdown(f"No benchmarks available in the '{category}' category.")
            continue

        columns_to_select = []
        if is_public:
            columns_to_select.append("Model Type")

        columns_to_select.extend(names_filtered)

        df_category = df_main[columns_to_select].apply(
            lambda col: col.map(
                lambda val: "N/A"
                if pd.isna(val)
                # Make sure that string columns are not converted
                else (val if type(val) is str else f"{val:.2f}")
            )
        )

        # Apply coloring and display
        st.dataframe(
            df_category.style.map(
                color_scores, subset=pd.IndexSlice[:, names_filtered]
            ),
        )
