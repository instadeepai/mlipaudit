import pandas as pd
import streamlit as st


@st.cache_data
def prepare_data(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
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
        total_score = 0.0
        num_benchmarks = 0
        for benchmark_name, score_value in benchmark_scores.items():
            if score_value is not None:
                row[benchmark_name] = score_value
                total_score += score_value
                num_benchmarks += 1
        row["Overall Score"] = total_score / num_benchmarks if num_benchmarks > 0 else 0
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


def leaderboard_page(
    scores: dict[str, dict[str, float]] | dict[str, dict[str, dict[str, float]]],
    remote: bool = False,
) -> None:
    """Leaderboard page. Takes the preprocessed scores and displays them.
    If remote is False, display all the results in a single table, otherwise
    for the remote leaderboard, separate the scores into two tables.

    Args:
        scores: The preprocessed scores.
        remote: Whether displaying locally or for the remote leaderboard.
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

    if remote:
        df_int, df_ext = (
            prepare_data(scores["internal"]),
            prepare_data(scores["external"]),
        )
        df_sorted_int = df_int.sort_values(by="Overall Score", ascending=False)
        df_sorted_ext = df_ext.sort_values(by="Overall Score", ascending=False)

        st.dataframe(
            df_sorted_int.style.map(_color_overall_score, subset=["Overall Score"]).map(
                _color_individual_score,
                subset=[
                    col
                    for col in df_sorted_int.columns
                    if col not in ["Model", "Overall Score"]
                ],
            ),
            hide_index=True,
        )
        st.dataframe(
            df_sorted_ext.style.map(_color_overall_score, subset=["Overall Score"]).map(
                _color_individual_score,
                subset=[
                    col
                    for col in df_sorted_ext.columns
                    if col not in ["Model", "Overall Score"]
                ],
            ),
            hide_index=True,
        )

    else:
        df = prepare_data(scores)
        df_sorted = df.sort_values(by="Overall Score", ascending=False)

        st.dataframe(
            df_sorted.style.map(_color_overall_score, subset=["Overall Score"]).map(
                _color_individual_score,
                subset=[
                    col for col in df.columns if col not in ["Model", "Overall Score"]
                ],
            ),
            hide_index=True,
        )
