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
import math
import statistics

from pydantic import BaseModel

from mlipaudit.benchmark import Benchmark, BenchmarkResult


class Metric(BaseModel):
    """A base model for a target metric.

    Attributes:
        name: The name of the target metric in the corresponding
            benchmarks' `Result` class.
        min_error: The minimum DFT threshold.
        max_error: The maximum DFT threshold.
    """

    name: str
    min_error: float
    max_error: float


METRIC_TARGETS_AND_TOLERANCES: dict[str, list[Metric]] = {
    "bond_length_distribution": [
        Metric(name="avg_fluctuation", min_error=0.005, max_error=0.05)
    ],
    "conformer_selection": [
        Metric(name="avg_mae", min_error=0.0, max_error=0.5),
        Metric(name="avg_rmse", min_error=0.0, max_error=1.5),
    ],
}


def compute_metric_score(value: float, threshold: float, alpha: float):
    """Compute the normalized score using a soft thresholding function
    given the DFT threshold. See Appendix B of 'MLIPAudit'.

    Arguments:
        value: The value of the metric.
        threshold: The maximum DFT threshold.
        alpha: The alpha parameter.

    Returns:
        The normalized score.
    """
    if value < threshold:
        return 1

    return math.exp(-alpha * (value - threshold) / threshold)


def compute_benchmark_score(
    result: BenchmarkResult, benchmark_class: type[Benchmark], alpha: float
):
    """Compute the score for a given benchmark.

    Arguments:
        result: The benchmark result object.
        benchmark_class: The class of the benchmark.
        alpha: The alpha normalization parameter.

    Returns:
        The average score across all metrics.
    """
    metric_scores = []
    metric_targets = METRIC_TARGETS_AND_TOLERANCES[benchmark_class.name]
    for metric in metric_targets:
        value = getattr(result, metric.name)
        metric_score = compute_metric_score(
            value, threshold=metric.max_error, alpha=alpha
        )
        metric_scores.append(metric_score)

    return statistics.mean(metric_scores)
