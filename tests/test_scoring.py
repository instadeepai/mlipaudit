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
import pytest
from numpy.testing import assert_allclose

from mlipaudit.benchmarks import ConformerSelectionResult
from mlipaudit.scoring import (
    compute_benchmark_score,
    compute_metric_score,
    compute_speed_score,
)


def test_compute_speed_score():
    """Test the Hill-function speed score."""
    midpoint, sharpness = 2.0, 1.0

    with pytest.raises(ValueError):
        compute_speed_score([1.0], midpoint=0.0, sharpness=sharpness)
    with pytest.raises(ValueError):
        compute_speed_score([1.0], midpoint=midpoint, sharpness=0.0)

    # value == midpoint -> 0.5; faster -> higher; None -> 0.
    scores = compute_speed_score([0.0, midpoint, None], midpoint, sharpness)
    assert_allclose(scores, np.array([1.0, 0.5, 0.0]))

    # Monotonically decreasing in the value.
    decreasing = compute_speed_score([1.0, 2.0, 4.0], midpoint, sharpness)
    assert decreasing[0] > decreasing[1] > decreasing[2]


def test_compute_metric_score():
    """Test compute_metric_score."""
    alpha = 0.0
    with pytest.raises(ValueError):
        value, threshold = 1.0, 3.0
        compute_metric_score(np.array([value]), threshold, alpha)

    with pytest.raises(ValueError):
        value, threshold = 1.0, 0.5
        assert compute_metric_score(np.array([value]), threshold, alpha) == 1.0

    alpha = 1.0
    value, threshold = 0.5, 1.0
    assert compute_metric_score(np.array([value]), threshold, alpha) == 1.0

    alpha = 3
    values, threshold = [0.8, 1.0, None], 0.5

    assert_allclose(
        compute_metric_score(np.array(values), threshold, alpha),
        np.array([0.16529889, 0.04978707, 0.0]),
    )


def test_compute_benchmark_score():
    """Test compute_benchmark_score."""
    conformer_selection_result = ConformerSelectionResult(
        avg_mae=0.1, avg_rmse=1.5, molecules=[]
    )

    alpha = 3.0
    score = compute_benchmark_score(
        [[conformer_selection_result.avg_mae], [conformer_selection_result.avg_rmse]],
        [0.5, 1.5],
    )
    assert type(score) is float
    assert score == 1.0

    conformer_selection_result = ConformerSelectionResult(
        avg_mae=0.1, avg_rmse=2.5, molecules=[]
    )

    score = compute_benchmark_score(
        [[conformer_selection_result.avg_mae], [conformer_selection_result.avg_rmse]],
        [0.5, 1.5],
    )
    assert [score] == list(0.5 + compute_metric_score(np.array([2.5]), 1.5, alpha) / 2)
