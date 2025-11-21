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
from pydantic import NonNegativeFloat, PositiveFloat
from streamlit.testing.v1 import AppTest

from mlipaudit.benchmark import Benchmark, BenchmarkResult
from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    ConformerSelectionBenchmark,
    DihedralScanBenchmark,
    FoldingStabilityBenchmark,
    NoncovalentInteractionsBenchmark,
    ReactivityBenchmark,
    ReferenceGeometryStabilityBenchmark,
    RingPlanarityBenchmark,
    SamplingBenchmark,
    ScalingBenchmark,
    SolventRadialDistributionBenchmark,
    SolventRadialDistributionResult,
    SolventRadialDistributionStructureResult,
    StabilityBenchmark,
    TautomersBenchmark,
    WaterRadialDistributionBenchmark,
)
from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    BondLengthDistributionMoleculeResult,
    BondLengthDistributionResult,
)
from mlipaudit.benchmarks.folding_stability.folding_stability import (
    FoldingStabilityMoleculeResult,
    FoldingStabilityResult,
)
from mlipaudit.benchmarks.reference_geometry_stability.reference_geometry_stability import (  # noqa: E501
    ReferenceGeometryStabilityDatasetResult,
    ReferenceGeometryStabilityResult,
)
from mlipaudit.benchmarks.ring_planarity.ring_planarity import (
    RingPlanarityMoleculeResult,
    RingPlanarityResult,
)
from mlipaudit.benchmarks.sampling.sampling import SamplingResult, SamplingSystemResult
from mlipaudit.benchmarks.water_radial_distribution.water_radial_distribution import (
    WaterRadialDistributionResult,
)
from mlipaudit.ui import (
    bond_length_distribution_page,
    conformer_selection_page,
    dihedral_scan_page,
    folding_stability_page,
    leaderboard_page,
    noncovalent_interactions_page,
    reactivity_page,
    reference_geometry_stability_page,
    ring_planarity_page,
    sampling_page,
    scaling_page,
    solvent_radial_distribution_page,
    stability_page,
    tautomers_page,
    water_radial_distribution_page,
)

BenchmarkResultForMultipleModels: TypeAlias = dict[str, BenchmarkResult]

DUMMY_SCORES_FOR_LEADERBOARD = {
    "model_1_int": {"overall_score": 0.75, "a": 0.7, "b": 0.8},
    "model_2_ext": {"overall_score": 0.5, "a": 0.3, "b": 0.7},
}


def _add_failed_molecule(
    benchmark_class,
    subresult_class,
    annotation_origin,
    name,
    kwargs_for_result,
    kwargs_for_subresult,
):
    # Create additional failed molecule
    failed_mol = None
    if "failed" in subresult_class.model_fields.keys():
        if benchmark_class in [
            BondLengthDistributionBenchmark,
            RingPlanarityBenchmark,
            ConformerSelectionBenchmark,
        ]:
            key_name = "molecule_name"
        elif benchmark_class in [
            FoldingStabilityBenchmark,
            SamplingBenchmark,
            SolventRadialDistributionBenchmark,
            StabilityBenchmark,
            ScalingBenchmark,
            NoncovalentInteractionsBenchmark,
        ]:
            key_name = "structure_name"
        elif benchmark_class in [DihedralScanBenchmark]:
            key_name = "fragment_name"

        elif benchmark_class in [TautomersBenchmark]:
            key_name = "structure_id"
        kwargs_for_failed = {key_name: "failed_mol", "failed": True}

        if benchmark_class is StabilityBenchmark:
            kwargs_for_failed.update({
                "description": "description",
                "num_steps": 1,
                "score": 0.0,
            })
        elif benchmark_class is ScalingBenchmark:
            kwargs_for_failed.update({
                "num_atoms": 10,
                "num_steps": 1,
                "num_episodes": 10,
            })
        elif benchmark_class is NoncovalentInteractionsBenchmark:
            kwargs_for_failed.update({
                "system_id": "id",
                "dataset": "dataset",
                "group": "group",
            })

        failed_mol = subresult_class(**kwargs_for_failed)

    if annotation_origin is list:
        kwargs_for_result[name] = [subresult_class(**kwargs_for_subresult)]  # type: ignore
        if failed_mol:
            kwargs_for_result[name].append(failed_mol)  # type: ignore
    else:
        kwargs_for_result[name] = {
            "test": subresult_class(**kwargs_for_subresult)  # type: ignore
        }
        if failed_mol:
            kwargs_for_result[name]["failed_mol"] = failed_mol  # type: ignore

    return kwargs_for_result


def _add_failed_model(benchmark_class, model_results) -> dict[str, BenchmarkResult]:
    if benchmark_class is BondLengthDistributionBenchmark:
        model_results["model_3"] = BondLengthDistributionResult(**{  # type: ignore
            "molecules": [
                BondLengthDistributionMoleculeResult(
                    molecule_name="failed_mol",
                    failed=True,
                )
            ],
            "failed": True,
            "score": 0.0,
        })
    elif benchmark_class is FoldingStabilityBenchmark:
        model_results["model_3"] = FoldingStabilityResult(**{  # type: ignore
            "molecules": [
                FoldingStabilityMoleculeResult(
                    structure_name="failed_mol",
                    failed=True,
                )
            ],
            "failed": True,
            "score": 0.0,
        })
    elif benchmark_class is RingPlanarityBenchmark:
        model_results["model_3"] = RingPlanarityResult(**{  # type: ignore
            "molecules": [
                RingPlanarityMoleculeResult(
                    molecule_name="failed_mol",
                    failed=True,
                )
            ],
            "failed": True,
            "score": 0.0,
        })
    elif benchmark_class is SamplingBenchmark:
        model_results["model_3"] = SamplingResult(**{
            "systems": [SamplingSystemResult(structure_name="failed_mol", failed=True)],
            "exploded_systems": ["failed_mol"],
            "failed": True,
            "score": 0.0,
        })
    elif benchmark_class is ReferenceGeometryStabilityBenchmark:
        dataset_result = ReferenceGeometryStabilityDatasetResult(
            rmsd_values=[None, None], num_exploded=2, num_bad_rmsds=0, failed=True
        )
        model_results["model_3"] = ReferenceGeometryStabilityResult(
            openff_charged=dataset_result,
            openff_neutral=dataset_result,
            failed=True,
            score=0.0,
        )
    elif benchmark_class is SolventRadialDistributionBenchmark:
        model_results["model_3"] = SolventRadialDistributionResult(
            structure_names=["failed_mol"],
            structures=[
                SolventRadialDistributionStructureResult(
                    structure_name="failed_mol", failed=True, score=0.0
                )
            ],
            failed=True,
            score=0.0,
        )
    elif benchmark_class is WaterRadialDistributionBenchmark:
        model_results["model_3"] = WaterRadialDistributionResult(failed=True, score=0.0)
    return model_results


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
            if field.annotation in [
                float,
                NonNegativeFloat,
                float | None,
                NonNegativeFloat | None,
            ]:
                kwargs_for_result[name] = 0.675
                continue

            if field.annotation in [dict[str, float], dict[str, float] | None]:
                kwargs_for_result[name] = {"test:test": 0.5}  # type: ignore
                continue

            if field.annotation in [list[str], list[str] | None]:
                kwargs_for_result[name] = ["test"]  # type: ignore
                continue

            if field.annotation in [list[float], list[float] | None]:
                kwargs_for_result[name] = [3.0, 4.0]  # type: ignore
                continue

            # Second, we handle some more specialized cases for some more
            # unique benchmarks
            if field.annotation == ReferenceGeometryStabilityDatasetResult:
                kwargs_for_result[name] = ReferenceGeometryStabilityDatasetResult(
                    rmsd_values=[1.0, None],
                    avg_rmsd=1.0,
                    num_exploded=1,
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
                    if subfield.annotation in [int, int | None]:
                        kwargs_for_subresult[subname] = 1
                    if subfield.annotation in [
                        float,
                        NonNegativeFloat,
                        PositiveFloat,
                        float | None,
                        NonNegativeFloat | None,
                        PositiveFloat | None,
                    ]:
                        kwargs_for_subresult[subname] = 0.4  # type: ignore
                    if subfield.annotation in [list[float], list[float] | None]:
                        kwargs_for_subresult[subname] = [0.3, 0.5]  # type: ignore
                    if subfield.annotation in [
                        dict[str, float],
                        dict[str, float] | None,
                    ]:
                        kwargs_for_subresult[subname] = {"test": 0.5}  # type: ignore
                    if subfield.annotation is str:
                        kwargs_for_subresult[subname] = "test"  # type: ignore

                        # special case
                        if benchmark_class is DihedralScanBenchmark:
                            kwargs_for_subresult[subname] = "fragment_001"  # type: ignore

                kwargs_for_result = _add_failed_molecule(
                    benchmark_class=benchmark_class,
                    subresult_class=subresult_class,
                    annotation_origin=annotation_origin,
                    name=name,
                    kwargs_for_result=kwargs_for_result,
                    kwargs_for_subresult=kwargs_for_subresult,
                )

        # Manually add the score for the test
        if benchmark_class not in [
            ScalingBenchmark,
            SolventRadialDistributionBenchmark,
        ]:
            kwargs_for_result["score"] = 0.3

        model_results = {
            "model_1": benchmark_class.result_class(**kwargs_for_result),  # type: ignore
            "model_2": benchmark_class.result_class(**kwargs_for_result),  # type: ignore
        }

        model_results = _add_failed_model(benchmark_class, model_results)

        return model_results

    return data_func


def _app_script(page_func, data_func, scores, is_public):
    import functools  # noqa

    import streamlit as st  # noqa

    from mlipaudit.ui.utils import model_selection  # noqa

    available_models = ["model_1", "model_2", "model_3"]
    model_selection(unique_model_names=available_models)

    if scores is None:  # Benchmark page
        _page_func = functools.partial(
            page_func,
            data_func=data_func,
        )
    else:  # Leaderboard page
        _page_func = functools.partial(
            page_func,
            scores=scores,
            is_public=is_public,
        )

    page = st.Page(
        _page_func,
        title="Page",
        url_path="page",
    )

    pages_to_show = [page]
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
        (ReferenceGeometryStabilityBenchmark, reference_geometry_stability_page),
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

    args_for_app = (page_to_test, dummy_data_func, None, None)
    app = AppTest.from_function(_app_script, args=args_for_app)

    app.run(timeout=10.0)
    assert not app.exception


@pytest.mark.parametrize("is_public", [True, False])
def test_leaderboard_page_is_working_correctly(is_public):
    """Tests the leaderboard UI page with the AppTest pattern from streamlit."""
    args_for_app = (leaderboard_page, None, DUMMY_SCORES_FOR_LEADERBOARD, is_public)
    app = AppTest.from_function(_app_script, args=args_for_app)

    app.run()
    assert not app.exception
