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

ALPHA = 1.0


def compute_metric_score(value: float, threshold: float, alpha: float):
    """Compute the normalized score using a soft thresholding function
    given the DFT threshold. See Appendix B of 'MLIPAudit'.

    Args:
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
    metric_values: list[float],
    thresholds: list[float],
):
    """Given a list of metric values and its associated list of acceptable thresholds,
    compute the benchmark score by taking the average of the normalized scores.

    Args:
        metric_values: The list of metric values.
        thresholds: The list of acceptable DFT thresholds.

    Returns:
        The benchmark score.
    """
    metric_scores = []
    for metric_value, threshold in zip(metric_values, thresholds):
        metric_score = compute_metric_score(metric_value, threshold, ALPHA)
        metric_scores.append(metric_score)

    return statistics.mean(metric_scores)
