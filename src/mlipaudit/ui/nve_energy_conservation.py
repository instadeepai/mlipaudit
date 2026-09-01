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

import altair as alt
import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import (
    NVEEnergyConservationBenchmark,
    NVEEnergyConservationResult,
)
from mlipaudit.ui.page_wrapper import UIPageWrapper
from mlipaudit.ui.utils import (
    display_failed_models,
    display_model_scores,
    fetch_selected_models,
    filter_failed_results,
    get_failed_models,
)

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[
    ModelName, NVEEnergyConservationResult
]


def _scores_dataframe(
    data: BenchmarkResultForMultipleModels, selected_models: list[str]
) -> pd.DataFrame:
    """Build the per-model summary score table.

    Args:
        data: Mapping from model name to its benchmark result.
        selected_models: The models currently selected in the sidebar.

    Returns:
        A DataFrame indexed by model name with a ``Score`` column.
    """
    rows = {
        model_name: {"Score": result.score}
        for model_name, result in data.items()
        if model_name in selected_models
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model name")


def _drift_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
    system_name: str,
) -> pd.DataFrame:
    """Build the long-format drift curve + linear fit data for one system.

    Args:
        data: Mapping from model name to its benchmark result.
        selected_models: The models currently selected in the sidebar.
        system_name: The system to extract drift curves for.

    Returns:
        A long-format DataFrame with columns ``Time (ps)``, ``Drift (eV)``,
        ``Model`` and ``Kind`` (either ``Drift`` or ``Linear fit``).
    """
    records = []
    for model_name, result in data.items():
        if model_name not in selected_models:
            continue
        for structure in result.structure_results:
            if structure.structure_name != system_name:
                continue
            if structure.skipped or structure.failed or len(structure.times_ps) < 2:
                continue
            slope = structure.drift_slope_ev_per_ps or 0.0
            intercept = structure.intercept_ev or 0.0
            for time_ps, drift in zip(structure.times_ps, structure.energy_drift_ev):
                records.append({
                    "Time (ps)": time_ps,
                    "Drift (eV)": drift,
                    "Model": model_name,
                    "Kind": "Drift",
                })
                records.append({
                    "Time (ps)": time_ps,
                    "Drift (eV)": slope * time_ps + intercept,
                    "Model": model_name,
                    "Kind": "Linear fit",
                })
    return pd.DataFrame(records)


def _system_metrics_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
    system_name: str,
) -> pd.DataFrame:
    """Build the per-model metrics table for one system.

    Args:
        data: Mapping from model name to its benchmark result.
        selected_models: The models currently selected in the sidebar.
        system_name: The system to extract per-model metrics for.

    Returns:
        A DataFrame indexed by model name with the drift metrics and score.
    """
    rows = {}
    for model_name, result in data.items():
        if model_name not in selected_models:
            continue
        for structure in result.structure_results:
            if structure.structure_name != system_name:
                continue
            if structure.skipped or structure.failed:
                continue
            rows[model_name] = {
                "Drift slope (eV/ps)": structure.drift_slope_ev_per_ps,
                "Total drift (eV)": structure.total_drift_ev,
                "Kinetic energy std (eV)": structure.kinetic_energy_std_ev,
                "Energy drift ratio": structure.energy_drift_ratio,
                "Score": structure.score,
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model name")


def nve_energy_conservation_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the NVE energy-conservation benchmark.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# NVE energy conservation")

    st.markdown(
        "This benchmark runs short microcanonical (NVE) molecular dynamics "
        "trajectories and measures how well an MLIP conserves the total mechanical "
        "energy, E(t) = PE(t) + KE(t). A good model keeps the total energy flat over "
        "the trajectory; a poorly conserving one shows a systematic drift. The "
        "headline metric is the magnitude of the fitted total-energy drift over the "
        "run, divided by the standard deviation of the kinetic energy, so that a "
        "smaller dimensionless ratio means better energy conservation."
    )

    st.markdown(
        "For more information, see the [docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/general/nve_energy_conservation.html)."
    )

    if "nve_energy_conservation_cached_data" not in st.session_state:
        st.session_state.nve_energy_conservation_cached_data = data_func()

    data: BenchmarkResultForMultipleModels = (
        st.session_state.nve_energy_conservation_cached_data
    )

    if not data:
        st.markdown("**No results to display**.")
        return

    selected_models = fetch_selected_models(available_models=list(data.keys()))

    if not selected_models:
        st.markdown("**No results to display**.")
        return

    failed_models = get_failed_models(data)
    display_failed_models(failed_models)
    data = filter_failed_results(data)

    if not data:
        st.markdown("**No results to display**.")
        return

    st.markdown("## Summary statistics")
    df_scores = _scores_dataframe(data, selected_models)
    if not df_scores.empty:
        df_scores.sort_values("Score", ascending=False, inplace=True)
        display_model_scores(df_scores)

    st.markdown("## Total-energy drift per system")
    st.markdown(
        "The solid line shows the total-energy drift relative to the first frame and "
        "the dashed line its linear fit. The flatter the curve, the better the model "
        "conserves energy."
    )

    available_systems = sorted({
        structure.structure_name
        for result in data.values()
        for structure in result.structure_results
        if not structure.skipped and not structure.failed and structure.times_ps
    })

    if not available_systems:
        st.markdown("**No drift curves to display**.")
        return

    system_name = st.selectbox(
        "Select a system",
        available_systems,
        format_func=lambda name: name.replace("_", " "),
    )

    df_drift = _drift_dataframe(data, selected_models, system_name)
    if df_drift.empty:
        st.markdown("**No drift curves to display for the selected models**.")
        return

    chart = (
        alt.Chart(df_drift)
        .mark_line()
        .encode(
            x=alt.X("Time (ps):Q", title="Time (ps)"),
            y=alt.Y("Drift (eV):Q", title="Total-energy drift ΔE (eV)"),
            color=alt.Color("Model:N", title="Model"),
            # Solid = drift, dashed = linear fit (explained in the text above). No
            # separate legend for this: a second stacked legend squeezes the
            # "Time (ps)" x-axis title out under Streamlit's container-width autosize.
            strokeDash=alt.StrokeDash("Kind:N", legend=None),
        )
        .properties(width=800, height=400)
    )
    st.altair_chart(chart, width="stretch")

    df_metrics = _system_metrics_dataframe(data, selected_models, system_name)
    if not df_metrics.empty:
        df_metrics.sort_values("Score", ascending=False, inplace=True)
        st.dataframe(
            df_metrics.style.format(precision=4),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.2f"
                )
            },
        )


class NVEEnergyConservationPageWrapper(UIPageWrapper):
    """Page wrapper for the NVE energy-conservation benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return nve_energy_conservation_page

    @classmethod
    def get_benchmark_class(cls) -> type[NVEEnergyConservationBenchmark]:  # noqa: D102
        return NVEEnergyConservationBenchmark
