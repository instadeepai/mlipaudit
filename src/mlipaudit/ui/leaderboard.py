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

from mlipaudit.benchmarks import BENCHMARK_CATEGORIES
from mlipaudit.ui.utils import (
    remove_model_name_extensions_and_capitalize_benchmark_names,
    split_scores,
)


@st.cache_data
def parse_scores_dict_into_df(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Parse the scores into a dataframe.

    Args:
        scores: The parsed scores dictionary.

    Returns:
        A dataframe with cols ordered as Model name, Overall
            score, with the remaining cols.
    """
    df_data = []
    for model_name, benchmark_scores in scores.items():
        row: dict[str, str | float] = {"Model": model_name}
        for benchmark_name, score_value in benchmark_scores.items():
            if score_value is not None:
                row[benchmark_name] = score_value
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Reorder columns
    cols = df.columns.tolist()
    # Model, score, etc.
    cols = cols[:1] + cols[-1:] + cols[1:-1]
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


def _group_score_df_by_benchmark_category(score_df: pd.DataFrame) -> pd.DataFrame:
    for category in BENCHMARK_CATEGORIES:
        names = [
            b.name.replace("_", " ").capitalize()
            for b in BENCHMARK_CATEGORIES[category]  # type: ignore
        ]
        names_filtered = [b for b in names if b in score_df.columns]

        score_df[category] = score_df[names_filtered].mean(axis=1)
        score_df = score_df.drop(columns=names_filtered)

    columns_in_order = [
        "Model",
        "Overall score",
        "Small Molecules",
        "Biomolecules",
        "Molecular Liquids",
        "General",
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
    st.sidebar.markdown("# MLIPAudit")

    st.markdown(
        """
        MLIPAudit is a Python tool for benchmarking and validating
        Machine Learning Interatomic Potentials (MLIP) models,
        specifically those written in mlip-jax. It aims to cover
        a wide range of use cases and difficulties, providing users
        with a comprehensive overview of the performance of their models.
        """
    )
    if is_public:
        scores_int, scores_ext = split_scores(scores)
        scores_int, scores_ext = (
            remove_model_name_extensions_and_capitalize_benchmark_names(scores_int),  # type: ignore
            remove_model_name_extensions_and_capitalize_benchmark_names(scores_ext),  # type: ignore
        )
        df_int, df_ext = (
            parse_scores_dict_into_df(scores_int),
            parse_scores_dict_into_df(scores_ext),
        )
        df_sorted_int = df_int.sort_values(by="Overall score", ascending=False).round(2)
        df_sorted_ext = df_ext.sort_values(by="Overall score", ascending=False).round(2)

        df_grouped_int = _group_score_df_by_benchmark_category(df_sorted_int)
        df_grouped_ext = _group_score_df_by_benchmark_category(df_sorted_ext)

        st.markdown("## InstaDeep models")
        st.dataframe(df_grouped_int, hide_index=True)

        st.markdown("## Community models")
        st.dataframe(df_grouped_ext, hide_index=True)

    else:
        scores = remove_model_name_extensions_and_capitalize_benchmark_names(scores)  # type: ignore
        df = parse_scores_dict_into_df(scores)
        df_sorted = df.sort_values(by="Overall score", ascending=False).round(2)
        df_grouped = _group_score_df_by_benchmark_category(df_sorted)
        st.dataframe(df_grouped, hide_index=True)
