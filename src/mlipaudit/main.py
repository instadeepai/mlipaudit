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

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from mlip.models import ForceField, Mace, Nequip, Visnet
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.model_io import load_model_from_zip

from mlipaudit.benchmark import Benchmark
from mlipaudit.bond_length_distribution import BondLengthDistributionBenchmark
from mlipaudit.conformer_selection import ConformerSelectionBenchmark
from mlipaudit.dihedral_scan import DihedralScanBenchmark
from mlipaudit.folding_stability import FoldingStabilityBenchmark
from mlipaudit.io import write_benchmark_result_to_disk
from mlipaudit.noncovalent_interactions import NoncovalentInteractionsBenchmark
from mlipaudit.reactivity import ReactivityBenchmark
from mlipaudit.ring_planarity import RingPlanarityBenchmark
from mlipaudit.sampling import SamplingBenchmark
from mlipaudit.scaling import ScalingBenchmark
from mlipaudit.small_molecule_minimization import SmallMoleculeMinimizationBenchmark
from mlipaudit.solvent_radial_distribution import SolventRadialDistributionBenchmark
from mlipaudit.stability import StabilityBenchmark
from mlipaudit.tautomers import TautomersBenchmark
from mlipaudit.water_radial_distribution import WaterRadialDistributionBenchmark

logger = logging.getLogger("mlipaudit")

BENCHMARKS = [
    ConformerSelectionBenchmark,
    TautomersBenchmark,
    NoncovalentInteractionsBenchmark,
    DihedralScanBenchmark,
    RingPlanarityBenchmark,
    SmallMoleculeMinimizationBenchmark,
    FoldingStabilityBenchmark,
    BondLengthDistributionBenchmark,
    SamplingBenchmark,
    WaterRadialDistributionBenchmark,
    SolventRadialDistributionBenchmark,
    ReactivityBenchmark,
    StabilityBenchmark,
    ScalingBenchmark,
]


def _parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="uv run mlipaudit",
        description="Runs a full benchmark with given models.",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="paths to the model zip archives",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="path to the output directory"
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        required=False,
        choices=["all"] + list(benchmark.name for benchmark in BENCHMARKS),
        default="all",
        help="List of benchmarks to run.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="run the benchmarks in fast-dev-run mode",
    )
    return parser


# TODO: We should probably handle this in a different (nicer) way
def _model_class_from_name(model_name: str) -> type[MLIPNetwork]:
    if "visnet" in model_name:
        return Visnet
    if "mace" in model_name:
        return Mace
    if "nequip" in model_name:
        return Nequip
    raise NotImplementedError(
        "Name of model zip archive does not contain info about the type of MLIP model."
    )


def _get_benchmarks_to_run(args: Namespace) -> list[type[Benchmark]]:
    if args.benchmarks == "all":
        return BENCHMARKS
    else:
        benchmarks_to_run = []
        for benchmark_class in BENCHMARKS:
            if benchmark_class.name in args.benchmarks:
                benchmarks_to_run.append(benchmark_class)
        return benchmarks_to_run


def _can_run_model_on_benchmark(
    benchmark_class: type[Benchmark], force_field: ForceField
) -> bool:
    """Checks that we can run a force field on a certain benchmark,
    logging whether the benchmark can run or not.

    Returns:
        Whether the benchmark can run or not.
    """
    if not benchmark_class.check_can_run_model(force_field):
        missing_atomic_species = benchmark_class.get_missing_atomic_species(force_field)
        logger.info(
            "Skipping benchmark %s due to missing species: %s",
            benchmark_class.name,
            missing_atomic_species,
        )
        return False

    logger.info("Running benchmark %s.", benchmark_class.name)
    return True


def main():
    """Main for the MLIPAudit benchmark."""
    args = _parser().parse_args()
    output_dir = Path(args.output)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        force=True,
    )
    logger.setLevel(logging.INFO)

    benchmarks_to_run = _get_benchmarks_to_run(args)

    for model in args.models:
        model_name = Path(model).stem
        logger.info("Running benchmark with model %s.", model_name)

        model_class = _model_class_from_name(model_name)
        force_field = load_model_from_zip(model_class, model)

        for benchmark_class in benchmarks_to_run:
            if not _can_run_model_on_benchmark(benchmark_class, force_field):
                continue
            benchmark = benchmark_class(
                force_field=force_field, fast_dev_run=args.fast_dev_run
            )
            benchmark.run_model()
            result = benchmark.analyze()

            write_benchmark_result_to_disk(
                benchmark_class.name, result, output_dir / model_name
            )
            logger.info(
                "Wrote benchmark result to disk at path %s.",
                output_dir / model_name / benchmark_class.name,
            )

    logger.info("Completed all benchmarks with all models.")
