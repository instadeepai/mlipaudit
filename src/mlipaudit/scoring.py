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

import numpy as np

ALPHA = 3.0


def compute_metric_score(
    values: np.ndarray, threshold: float, alpha: float
) -> np.ndarray:
    """Compute the normalized score for an array of values using a soft
    thresholding function given a max. desired deviation threshold.

    Args:
        values: A NumPy array of metric values.
        threshold: The maximum threshold accepted for each value.
        alpha: The alpha parameter. Must be a positive float.

    Returns:
        A NumPy array of normalized scores.

    Raises:
        ValueError: If alpha is not positive.
    """
    if alpha <= 0:
        raise ValueError("alpha must be a positive number.")

    scores = np.ones_like(values, dtype=float)

    # Find where value >= threshold
    above_threshold_indices = values >= threshold

    # Apply the exponential formula only to those values
    scores[above_threshold_indices] = np.exp(
        -alpha * (values[above_threshold_indices] - threshold) / threshold
    )
    return scores


def compute_benchmark_score(
    errors: list[list[float]],
    thresholds: list[float],
) -> float:
    """Given a list of metric values and its associated list of acceptable thresholds,
    compute the benchmark score by taking the average of the normalized scores.

    Args:
        errors: The list of metric values.
        thresholds: The list of acceptable max. thresholds.

    Returns:
        The benchmark score.
    """
    metric_scores = []
    for metric_errors, threshold in zip(errors, thresholds):
        scores = compute_metric_score(np.array(metric_errors), threshold, ALPHA)
        metric_scores.append(scores.mean())

    return float(np.mean(np.array(metric_scores)))
