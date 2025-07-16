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

import argparse
import logging
from pathlib import Path

from mlip.models import Mace, Nequip, Visnet
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.model_io import load_model_from_zip

from mlipaudit.io import write_benchmark_results_to_disk
from mlipaudit.small_molecule_conformer_selection import ConformerSelectionBenchmark

logger = logging.getLogger("mlipaudit")

BENCHMARKS = [ConformerSelectionBenchmark]


def _parser():
    parser = argparse.ArgumentParser(
        prog="python main.py",
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


def main():
    """Main for the MLIPAudit benchmark."""
    args = _parser().parse_args()
    output_dir = Path(args.output)

    for model in args.models:
        model_name = Path(model).stem
        logger.info("Running benchmark with model %s.", model_name)

        model_class = _model_class_from_name(model_name)
        force_field = load_model_from_zip(model_class, model)

        benchmarks = []
        for benchmark_class in BENCHMARKS:
            benchmarks.append(benchmark_class(force_field, fast_dev_run=False))

        results = {}

        for benchmark in benchmarks:
            benchmark.run_model()
            result = benchmark.analyze()
            results[benchmark.name] = result

        write_benchmark_results_to_disk(results, output_dir / model_name)


if __name__ == "__main__":
    main()
