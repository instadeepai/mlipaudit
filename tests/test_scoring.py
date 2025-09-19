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

import pytest

from mlipaudit.benchmarks import ConformerSelectionResult
from mlipaudit.scoring import compute_benchmark_score, compute_metric_score


def test_compute_metric_score():
    """Test compute_metric_score."""
    alpha = 0.0
    value, threshold = 1.0, 3.0
    assert compute_metric_score(value, threshold, alpha) == 1.0

    value, threshold = 1.0, 0.5
    assert compute_metric_score(value, threshold, alpha) == 1.0

    alpha = 1.0
    value, threshold = 0.5, 1.0
    assert compute_metric_score(value, threshold, alpha) == 1.0

    value, threshold = 1.0, 0.5
    assert compute_metric_score(value, threshold, alpha) == pytest.approx(0.367879441)


def test_compute_benchmark_score():
    """Test compute_benchmark_score."""
    conformer_selection_result = ConformerSelectionResult(
        avg_mae=0.1, avg_rmse=1.5, molecules=[]
    )

    alpha = 1.0
    score = compute_benchmark_score(
        [conformer_selection_result.avg_mae, conformer_selection_result.avg_rmse],
        [0.5, 1.5],
    )
    assert type(score) is float
    assert score == 1.0

    conformer_selection_result = ConformerSelectionResult(
        avg_mae=0.1, avg_rmse=2.5, molecules=[]
    )

    score = compute_benchmark_score(
        [conformer_selection_result.avg_mae, conformer_selection_result.avg_rmse],
        [0.5, 1.5],
    )
    assert score == 0.5 + compute_metric_score(2.5, 1.5, alpha) / 2
