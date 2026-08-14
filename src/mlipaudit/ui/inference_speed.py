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
import numpy as np
import pandas as pd
import streamlit as st

from mlipaudit.benchmarks import InferenceSpeedBenchmark, InferenceSpeedResult
from mlipaudit.benchmarks.inference_speed.inference_speed import (
    ASE_BACKEND,
    JAX_MD_BACKEND,
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
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, InferenceSpeedResult]

#: ns/day = timestep_fs * NS_PER_DAY_FACTOR / step_time_seconds.
NS_PER_DAY_FACTOR = 0.0864
#: Used when older results do not store the MD timestep (all past runs used 1 fs).
DEFAULT_TIMESTEP_FS = 1.0


def _atoms_per_s(time_s: float, num_atoms: int, timestep_fs: float) -> float:
    return num_atoms / time_s


def _ns_per_day(time_s: float, num_atoms: int, timestep_fs: float) -> float:
    return timestep_fs * NS_PER_DAY_FACTOR / time_s


#: Selectable y-axis metrics. ``family`` is "model" (engine-independent forward pass)
#: or "md" (end-to-end); "md" metrics also carry a ``backend`` ("ase"/"jax_md").
#: ``value`` maps (time_s, num_atoms, timestep_fs) -> displayed value.
METRICS: dict[str, dict] = {
    "Model throughput (atoms/s)": {
        "family": "model",
        "value": _atoms_per_s,
        "format": ".0f",
    },
    "Model forward time (s)": {
        "family": "model",
        "value": lambda time_s, num_atoms, timestep_fs: time_s,
        "format": ".4f",
    },
    "MD throughput — ASE (ns/day)": {
        "family": "md",
        "backend": ASE_BACKEND,
        "value": _ns_per_day,
        "format": ".1f",
    },
    "MD throughput — JAX-MD (ns/day)": {
        "family": "md",
        "backend": JAX_MD_BACKEND,
        "value": _ns_per_day,
        "format": ".1f",
    },
    "MD step time — ASE (s)": {
        "family": "md",
        "backend": ASE_BACKEND,
        "value": lambda time_s, num_atoms, timestep_fs: time_s,
        "format": ".4f",
    },
    "MD step time — JAX-MD (s)": {
        "family": "md",
        "backend": JAX_MD_BACKEND,
        "value": lambda time_s, num_atoms, timestep_fs: time_s,
        "format": ".4f",
    },
}


#: Metric used for the summary table, so that it does not depend on the selector. The
#: forward pass is engine-independent and is what the score is computed from.
SUMMARY_METRIC = "Model throughput (atoms/s)"


def _structure_time_and_samples(structure, spec: dict) -> tuple:
    """Return ``(central_time_s, [sample_times_s])`` for the metric spec.

    The samples are used for error bars. Returns ``(None, [])`` if the structure has
    no measurement for that metric (e.g. a backend that was not run / failed).
    """
    if spec["family"] == "model":
        return structure.average_forward_time, list(structure.forward_times)

    backend = structure.md.get(spec["backend"])
    if backend is None:
        return None, []
    return backend.average_step_time, list(backend.step_time_samples)


def _process_data_into_dataframe(
    data: BenchmarkResultForMultipleModels,
    selected_models: list[str],
    metric_name: str,
) -> pd.DataFrame:
    """Build a per-structure dataframe with the selected metric and its variance.

    Args:
        data: The benchmark results per model.
        selected_models: The models to include.
        metric_name: The metric to compute (a key of `METRICS`).

    Returns:
        A dataframe with one row per (model, structure).
    """
    spec = METRICS[metric_name]
    value_fn = spec["value"]
    df_data = []
    for model_name, result in data.items():
        if model_name not in selected_models:
            continue
        cutoff = getattr(result, "graph_cutoff_angstrom", None)
        for structure in result.structures:
            time_s, samples = _structure_time_and_samples(structure, spec)
            if time_s is None or time_s <= 0:
                continue

            timestep_fs = structure.timestep_fs or DEFAULT_TIMESTEP_FS
            num_atoms = structure.num_atoms
            metric_value = value_fn(time_s, num_atoms, timestep_fs)

            # Per-sample metric values give the spread shown as error bars. Results
            # without per-sample times simply have no error bar.
            sample_values = [
                value_fn(s, num_atoms, timestep_fs) for s in samples if s > 0
            ]
            std = float(np.std(sample_values)) if sample_values else 0.0

            df_data.append({
                "Model name": model_name,
                "Structure": structure.structure_name,
                "Num atoms": num_atoms,
                "Time (s)": time_s,
                "Graph cutoff (Å)": cutoff,
                metric_name: metric_value,
                "Metric low": max(metric_value - std, metric_value * 1e-3),
                "Metric high": metric_value + std,
            })
    return pd.DataFrame(df_data)


def _fit_scaling(num_atoms: np.ndarray, step_times: np.ndarray) -> tuple[float, float]:
    """Fit a power law ``step_time = a * N^k`` in log-log space.

    Args:
        num_atoms: System sizes.
        step_times: Average step times (s).

    Returns:
        A tuple ``(exponent_k, r_squared)``.
    """
    log_n, log_t = np.log(num_atoms), np.log(step_times)
    k, log_a = np.polyfit(log_n, log_t, 1)
    predicted = log_a + k * log_n
    ss_res = float(np.sum((log_t - predicted) ** 2))
    ss_tot = float(np.sum((log_t - log_t.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(k), r_squared


def _build_fit_lines(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Build smooth fitted scaling curves (in the selected metric) per model.

    Args:
        df: The per-structure dataframe.
        metric_name: The metric being plotted.

    Returns:
        A dataframe of fitted curve points, empty if no model has >= 2 sizes.
    """
    value_fn = METRICS[metric_name]["value"]
    rows = []
    for model_name, group in df.groupby("Model name"):
        if group["Num atoms"].nunique() < 2:
            continue
        num_atoms = group["Num atoms"].to_numpy(dtype=float)
        times = group["Time (s)"].to_numpy(dtype=float)
        k, log_a = np.polyfit(np.log(num_atoms), np.log(times), 1)
        grid = np.geomspace(num_atoms.min(), num_atoms.max(), num=50)
        fitted_times = np.exp(log_a) * grid**k
        for n, fitted_time in zip(grid, fitted_times):
            rows.append({
                "Model name": model_name,
                "Num atoms": n,
                metric_name: value_fn(fitted_time, n, DEFAULT_TIMESTEP_FS),
            })
    return pd.DataFrame(rows)


def _summary_table(
    df: pd.DataFrame, metric_name: str, scores: dict[str, float | None]
) -> pd.DataFrame:
    """Build the per-model summary (score, scaling exponent, fit quality, throughput).

    Args:
        df: The per-structure dataframe.
        metric_name: The metric being plotted (used for the headline value column).
        scores: The benchmark score per model name.

    Returns:
        A summary dataframe indexed by model name, with a ``Score`` column.
    """
    # The metric column is pre-formatted because `display_model_scores` renders the
    # whole table with a single precision, which does not suit all metrics.
    value_format = "{:" + METRICS[metric_name]["format"] + "}"
    rows = []
    for model_name, group in df.groupby("Model name"):
        largest = group.loc[group["Num atoms"].idxmax()]
        num_atoms = group["Num atoms"].to_numpy(dtype=float)
        times = group["Time (s)"].to_numpy(dtype=float)
        if group["Num atoms"].nunique() >= 2:
            exponent, r_squared = _fit_scaling(num_atoms, times)
            exponent_str = f"{exponent:.2f}"
            r_squared_str = f"{r_squared:.3f}"
        else:
            exponent_str, r_squared_str = "N/A", "N/A"

        rows.append({
            "Model name": model_name,
            "Score": scores.get(model_name),
            "Graph cutoff (Å)": group["Graph cutoff (Å)"].iloc[0],
            "Scaling exponent (time ∝ Nᵏ)": exponent_str,
            "R²": r_squared_str,
            f"{metric_name} @ largest system": value_format.format(
                largest[metric_name]
            ),
            "Largest system (atoms)": int(largest["Num atoms"]),
        })
    return pd.DataFrame(rows).set_index("Model name")


def plot_all_models_performance(
    df: pd.DataFrame, metric_name: str, log_scale: bool
) -> alt.Chart:
    """Plot the inference-speed curves for all models together.

    Args:
        df: The per-structure dataframe.
        metric_name: The metric to plot on the y-axis (a key of `METRICS`).
        log_scale: Whether to use log-log axes.

    Returns:
        The Altair chart.
    """
    scale_type = "log" if log_scale else "linear"
    value_format = METRICS[metric_name]["format"]
    color = alt.Color("Model name:N", title="Model", legend=alt.Legend(title="Model"))
    x = alt.X(
        "Num atoms:Q",
        title="System size (number of atoms)",
        scale=alt.Scale(type=scale_type, zero=False),
    )

    points = (
        alt.Chart(df)
        .mark_point(size=70, filled=True, opacity=0.85)
        .encode(
            x=x,
            y=alt.Y(
                f"{metric_name}:Q",
                title=metric_name,
                scale=alt.Scale(type=scale_type, zero=False),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("Model name:N", title="Model"),
                alt.Tooltip("Structure:N", title="Structure"),
                alt.Tooltip("Num atoms:Q", title="Number of atoms"),
                alt.Tooltip(f"{metric_name}:Q", title=metric_name, format=value_format),
            ],
        )
    )

    error_bars = (
        alt.Chart(df)
        .mark_rule(opacity=0.6)
        .encode(
            x=x,
            y=alt.Y("Metric low:Q", title=metric_name),
            y2="Metric high:Q",
            color=color,
        )
    )

    layers = [error_bars, points]

    fit_df = _build_fit_lines(df, metric_name)
    if not fit_df.empty:
        lines = (
            alt.Chart(fit_df)
            .mark_line(strokeWidth=2)
            .encode(x=x, y=alt.Y(f"{metric_name}:Q", title=metric_name), color=color)
        )
        layers.append(lines)

    chart = alt.layer(*layers).properties(width=800, height=500).interactive()
    st.altair_chart(chart, use_container_width=True)
    return chart


def inference_speed_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the inference-speed page.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Inference speed")

    st.markdown(
        "This module assesses how fast MLIPs run, across several systems of varying "
        "size. Two complementary speeds are reported: **model throughput** (the raw "
        "forward pass, engine-independent) and **MD throughput** (end-to-end, with the "
        "simulation engine). Switch between them with the metric selector; the gap "
        "between the two reflects simulation overhead."
    )

    st.markdown(
        "For more information, see the "
        "[docs](https://instadeepai.github.io/mlipaudit"
        "/benchmarks/general/inference_speed.html)."
    )

    if "inference_speed_data" not in st.session_state:
        st.session_state.inference_speed_data = data_func()

    data = st.session_state.inference_speed_data

    if not data:
        st.markdown("**No results to display**.")
        return

    failed_models = get_failed_models(data)
    display_failed_models(failed_models)
    data = filter_failed_results(data)

    selected_models = fetch_selected_models(available_models=list(data.keys()))

    if not selected_models:
        st.markdown("**No results to display**.")
        return

    st.markdown("## Summary statistics")

    df_forward = _process_data_into_dataframe(data, selected_models, SUMMARY_METRIC)

    if df_forward.empty:
        st.markdown("**No results to display**.")
        return

    scores = {model_name: result.score for model_name, result in data.items()}
    df_summary = _summary_table(df_forward, SUMMARY_METRIC, scores)
    df_summary.sort_values("Score", ascending=False, inplace=True)
    display_model_scores(df_summary)

    st.caption(
        "The score rewards fast models: each system contributes a Hill-function score "
        "on its model forward time relative to the reference time for a system of that "
        "size, and the benchmark score is the mean over systems. It is based on the "
        "forward pass (not the MD step) so that it does not depend on the simulation "
        "engine."
    )

    st.markdown("## Inference speed: throughput vs system size")

    col_metric, col_scale = st.columns([3, 1])
    with col_metric:
        metric_name = st.selectbox("Metric", options=list(METRICS.keys()), index=0)
    with col_scale:
        log_scale = st.checkbox("Log–log axes", value=True)

    df = _process_data_into_dataframe(data, selected_models, metric_name)

    if df.empty:
        st.markdown("**No results to display**.")
        return

    plot_all_models_performance(df, metric_name, log_scale)

    st.caption(
        "Points are per-system measurements (error bars show the spread across "
        "repeats); lines are power-law fits. For external (ASE) models the model "
        "metric includes neighbour-list construction, whereas for mlip models it is "
        "the pure network forward. All times are wall-clock and hardware-relative — "
        "only compare models run on the same GPU."
    )


class InferenceSpeedPageWrapper(UIPageWrapper):
    """Page wrapper for the inference-speed benchmark."""

    @classmethod
    def get_page_func(  # noqa: D102
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        return inference_speed_page

    @classmethod
    def get_benchmark_class(cls) -> type[InferenceSpeedBenchmark]:  # noqa: D102
        return InferenceSpeedBenchmark
