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
from typing import Callable, TypeAlias, get_args, get_origin

import pytest
from streamlit.testing.v1 import AppTest

from mlipaudit.benchmark import Benchmark, BenchmarkResult
from mlipaudit.bond_length_distribution import BondLengthDistributionBenchmark
from mlipaudit.conformer_selection import ConformerSelectionBenchmark
from mlipaudit.dihedral_scan import DihedralScanBenchmark
from mlipaudit.folding_stability import FoldingStabilityBenchmark
from mlipaudit.noncovalent_interactions import NoncovalentInteractionsBenchmark
from mlipaudit.reactivity import ReactivityBenchmark
from mlipaudit.ring_planarity import RingPlanarityBenchmark
from mlipaudit.sampling import SamplingBenchmark
from mlipaudit.scaling import ScalingBenchmark
from mlipaudit.small_molecule_minimization import SmallMoleculeMinimizationBenchmark
from mlipaudit.small_molecule_minimization.small_molecule_minimization import (
    SmallMoleculeMinimizationDatasetResult,
)
from mlipaudit.solvent_radial_distribution import SolventRadialDistributionBenchmark
from mlipaudit.stability import StabilityBenchmark
from mlipaudit.tautomers import TautomersBenchmark
from mlipaudit.ui import (
    bond_length_distribution_page,
    conformer_selection_page,
    dihedral_scan_page,
    folding_stability_page,
    leaderboard_page,
    noncovalent_interactions_page,
    reactivity_page,
    ring_planarity_page,
    sampling_page,
    scaling_page,
    small_molecule_minimization_page,
    solvent_radial_distribution_page,
    stability_page,
    tautomers_page,
    water_radial_distribution_page,
)
from mlipaudit.water_radial_distribution import WaterRadialDistributionBenchmark

BenchmarkResultForMultipleModels: TypeAlias = dict[str, BenchmarkResult]

DUMMY_SCORES_FOR_LEADERBOARD = {
    "model_1_int": {"overall_score": 0.75, "a": 0.7, "b": 0.8},
    "model_2_ext": {"overall_score": 0.5, "a": 0.3, "b": 0.7},
}


# Important note:
# ---------------
# The following function acts a generic way to artificially populate benchmark results
# classes with dummy data. The purpose of it is that we can easily get dummy data
# for each benchmark without having to specify it manually for each. When adding a new
# benchmark to the test below, make sure that the dummy data for that benchmark works
# and otherwise modify this function to handle that case, potentially, by just
# adding a special case if it would otherwise break other cases.
def _construct_data_func_for_benchmark(
    benchmark_class: type[Benchmark],
) -> Callable[[], BenchmarkResultForMultipleModels]:
    def data_func() -> BenchmarkResultForMultipleModels:
        kwargs_for_result = {}
        for name, field in benchmark_class.result_class.model_fields.items():  # type: ignore
            # First, we handle some standard cases
            if field.annotation is float:
                kwargs_for_result[name] = 0.675
                continue

            if field.annotation == dict[str, float]:
                kwargs_for_result[name] = {"test:test": 0.5}  # type: ignore
                continue

            if field.annotation == list[str]:
                kwargs_for_result[name] = ["test"]  # type: ignore
                continue

            if field.annotation == list[float]:
                kwargs_for_result[name] = [3.0, 4.0]  # type: ignore
                continue

            # Second, we handle some more specialized cases for some more
            # unique benchmarks
            if field.annotation == SmallMoleculeMinimizationDatasetResult:
                kwargs_for_result[name] = SmallMoleculeMinimizationDatasetResult(
                    rmsd_values=[1.0, 2.0],
                    avg_rmsd=1.5,
                    num_exploded=0,
                    num_bad_rmsds=0,
                )
                continue

            # Lastly, we have in most benchmarks a list or a dictionary that contains
            # subresult classes. We will populate those now:
            annotation_origin = get_origin(field.annotation)
            if annotation_origin in [list, dict]:
                idx = 0 if annotation_origin is list else 1
                subresult_class = get_args(field.annotation)[idx]
                kwargs_for_subresult = {}
                for subname, subfield in subresult_class.model_fields.items():
                    if subfield.annotation is int:
                        kwargs_for_subresult[subname] = 1
                    if subfield.annotation is float:
                        kwargs_for_subresult[subname] = 0.4  # type: ignore
                    if subfield.annotation == list[float]:
                        kwargs_for_subresult[subname] = [0.3, 0.5]  # type: ignore
                    if subfield.annotation == dict[str, float]:
                        kwargs_for_subresult[subname] = {"test": 0.5}  # type: ignore
                    if subfield.annotation is str:
                        kwargs_for_subresult[subname] = "test"  # type: ignore

                        # special case
                        if benchmark_class is DihedralScanBenchmark:
                            kwargs_for_subresult[subname] = "fragment_001"  # type: ignore

                if annotation_origin is list:
                    kwargs_for_result[name] = [subresult_class(**kwargs_for_subresult)]  # type: ignore
                else:
                    kwargs_for_result[name] = {
                        "test": subresult_class(**kwargs_for_subresult)  # type: ignore
                    }
        # Manually add the score for the test
        if benchmark_class not in [
            ScalingBenchmark,
            SolventRadialDistributionBenchmark,
        ]:
            kwargs_for_result["score"] = 0.3

        return {
            "model_1": benchmark_class.result_class(**kwargs_for_result),  # type: ignore
            "model_2": benchmark_class.result_class(**kwargs_for_result),  # type: ignore
        }

    return data_func


def _app_script(page_func, data_func, leaderboard_page_func, scores, is_public):
    import functools  # noqa

    import streamlit as st  # noqa

    leaderboard = st.Page(
        functools.partial(leaderboard_page_func, scores=scores, is_public=is_public),
        title="Leaderboard",
    )

    page = st.Page(
        functools.partial(
            page_func,
            data_func=data_func,
        ),
        title="Benchmark",
        url_path="benchmark",
    )

    pages_to_show = [leaderboard, page]
    pg = st.navigation(pages_to_show)
    pg.run()


@pytest.mark.parametrize(
    "benchmark_to_test, page_to_test",
    [
        (RingPlanarityBenchmark, ring_planarity_page),
        (ReactivityBenchmark, reactivity_page),
        (ConformerSelectionBenchmark, conformer_selection_page),
        (BondLengthDistributionBenchmark, bond_length_distribution_page),
        (FoldingStabilityBenchmark, folding_stability_page),
        (NoncovalentInteractionsBenchmark, noncovalent_interactions_page),
        (SmallMoleculeMinimizationBenchmark, small_molecule_minimization_page),
        (SolventRadialDistributionBenchmark, solvent_radial_distribution_page),
        (StabilityBenchmark, stability_page),
        (TautomersBenchmark, tautomers_page),
        (WaterRadialDistributionBenchmark, water_radial_distribution_page),
        (ScalingBenchmark, scaling_page),
        (SamplingBenchmark, sampling_page),
        (DihedralScanBenchmark, dihedral_scan_page),
    ],
)
def test_ui_page_is_working_correctly(benchmark_to_test, page_to_test):
    """Tests a UI page with dummy data and the AppTest pattern from streamlit."""
    dummy_data_func = _construct_data_func_for_benchmark(benchmark_to_test)

    # Testing this case for one benchmark is enough
    is_public = benchmark_to_test == RingPlanarityBenchmark

    args_for_app = (
        page_to_test,
        dummy_data_func,
        leaderboard_page,
        DUMMY_SCORES_FOR_LEADERBOARD,
        is_public,
    )
    app = AppTest.from_function(_app_script, args=args_for_app)

    app.run()
    assert not app.exception
