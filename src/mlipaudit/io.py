import json
import os
from pathlib import Path

from mlipaudit.benchmark import BenchmarkResult


def write_benchmark_results_to_disk(
    results: dict[str, BenchmarkResult | list[BenchmarkResult]],
    output_dir: str | os.PathLike,
) -> None:
    """Writes a collection of benchmark results to disk.

    Args:
        results: The results as a dictionary with the benchmark names as keys.
        output_dir: Directory to which to write the results.
    """
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True)

    for name, result in results.items():
        (_output_dir / name).mkdir(exist_ok=True)
        with (_output_dir / name / "result.json").open("w") as json_file:
            if type(result) is list:
                json_as_str = [json.loads(item.model_dump_json()) for item in result]
            else:
                json_as_str = json.loads(result.model_dump_json())  # type: ignore
            json.dump(json_as_str, json_file, indent=4)
