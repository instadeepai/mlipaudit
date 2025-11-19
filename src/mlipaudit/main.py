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
import textwrap
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

import mlipaudit
from mlipaudit.app import launch_app
from mlipaudit.benchmark import Benchmark
from mlipaudit.benchmarks import (
    BENCHMARK_NAMES,
    BENCHMARKS,
)
from mlipaudit.benchmarks_cli import run_benchmarks
from mlipaudit.run_mode import RunMode

logger = logging.getLogger("mlipaudit")

EXTERNAL_MODEL_VARIABLE_NAME = "mlipaudit_external_model"
DESCRIPTION = textwrap.dedent(f"""\
mlipaudit - mlip benchmarking suite. [version {mlipaudit.__version__}]

mlipaudit is a tool for rigorously evaluating machine learning
interatomic potentials across a wide range of chemical and
physical properties. It aims to cover a wide range of use cases
and difficulties, providing users with a comprehensive overview
of the performance of their models.

Run "mlipaudit benchmark -h" for help with running benchmarks,
and "mlipaudit gui -h" for help with the launching the GUI for
visualization of benchmark results.

For more advanced usage and detailed benchmark information, see
the documentation at https://instadeepai.github.io/mlipaudit/.

Examples:

    $ mlipaudit benchmark -m model1.zip model2.zip -o results/
    $ mlipaudit benchmark -m potential.zip -o output/ --benchmarks conformer_selection
    $ mlipaudit benchmark -m my_model.py -o output/
    $ mlipaudit gui results/
""")

EPILOG = textwrap.dedent("""\
For more information and detailed options, consult the official
documentation or visit our GitHub repository.
""")


def _subparse_benchmark(parser):
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="paths to the model zip archives or python files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="path to the output directory;"
        " will overwrite existing results for a given model",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default="./data",
        help="path to the input data directory",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        required=False,
        choices=["all"] + list(benchmark.name for benchmark in BENCHMARKS),
        default=["all"],
        help=f"list of benchmarks to run; defaults to all benchmarks;"
        f" mutually exclusive with '-e'; allowed values are:"
        f" {', '.join(['all'] + BENCHMARK_NAMES)}",
        metavar="",
    )
    group.add_argument(
        "-e",
        "--exclude-benchmarks",
        nargs="+",
        choices=list(b.name for b in BENCHMARKS),
        help=f"list of benchmarks to exclude; mutually exclusive with '-b';"
        f" allowed values are: {', '.join(BENCHMARK_NAMES)}",
        metavar="",
    )
    parser.add_argument(
        "-rm",
        "--run-mode",
        required=False,
        choices=[mode.value for mode in RunMode],
        default=RunMode.STANDARD.value,
        help="mode to run the benchmarks in, either 'dev', 'fast' or 'standard'",
        metavar="",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose logging output from the mlip library",
    )
    parser.add_argument(
        "-lt",
        "--log-timings",
        action="store_true",
        help="log the timings for each benchmark",
    )


def _subparse_app(parser):
    parser.add_argument(
        "results_dir",
        help="path to the results directory containing benchmark results",
    )
    parser.add_argument(
        "--is-public",
        action="store_true",
        help="whether the GUI app is launched in our public HuggingFace setting, "
        "which leads to a differently designed landing page",
    )


def _parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="mlipaudit",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + mlipaudit.__version__
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    # Create the 'benchmark' command
    parser_benchmark = subparsers.add_parser("benchmark", help="run benchmarks")

    _subparse_benchmark(parser_benchmark)

    # Create the 'app' command
    parser_gui = subparsers.add_parser(
        "gui",
        help="launch the mlipaudit web application",
    )

    _subparse_app(parser_gui)

    return parser


def _validate_benchmark_names(benchmark_names: list[str]) -> None:
    for benchmark_name in benchmark_names:
        if benchmark_name not in BENCHMARK_NAMES:
            raise ValueError(f"Invalid benchmark name: {benchmark_name}")


def _get_benchmarks_to_run(args: Namespace) -> list[type[Benchmark]]:
    if args.exclude_benchmarks is not None:
        _validate_benchmark_names(args.exclude_benchmarks)
        return [b for b in BENCHMARKS if b.name not in args.exclude_benchmarks]  # type: ignore
    elif "all" in args.benchmarks:
        return BENCHMARKS
    else:
        _validate_benchmark_names(args.benchmarks)
        return [b for b in BENCHMARKS if b.name in args.benchmarks]  # type: ignore


def main():
    """Main function for the mlipaudit command line interface."""
    parser = _parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        force=True,
    )

    if getattr(args, "verbose", False):
        logger.setLevel(logging.INFO)
    else:
        mlip_logger = logging.getLogger("mlip")
        mlip_logger.setLevel(logging.WARNING)

    if args.command == "benchmark":
        benchmarks_to_run = _get_benchmarks_to_run(args)
        run_benchmarks(
            model_paths=args.models,
            benchmarks_to_run=benchmarks_to_run,
            run_mode=args.run_mode,
            output_dir=args.output,
            data_input_dir=args.input,
            verbose=args.verbose,
            log_timings=args.log_timings,
        )
    elif args.command == "gui":
        launch_app(args.results_dir, args.is_public)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
