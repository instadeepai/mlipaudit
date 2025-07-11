"""ABC definition defining common benchmarking interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from mlip.models import ForceField
from mlip.models.predictor import Prediction
from mlip.simulation import SimulationState

from pathlib import Path

from pydantic import BaseModel, Field

class BaseBenchmarkResult(BaseModel):
    """A base model for all benchmark results."""

    benchmark_name: str = Field(..., description="The unique name of the associated benchmark.")


class Benchmark(ABC):
    """An Abstract Base Class for structuring MLIP benchmark calculations.

    This class uses the Template Method pattern. The public `run()` method
    defines the skeleton of the benchmark workflow, calling the abstract
    steps in a fixed order. Each concrete benchmark must implement the
    `_generate_data`, `_analyze_results`, and `_report` methods.

    Attributes:
        mlip: The MLIP to be benchmarked.
        output_dir: The output directory of the benchmark.
        results: The results of the benchmark.
    """

    def __init__(self, mlip: ForceField, output_dir: str | Path = "benchmark_results") -> None:
        """Initializes the benchmark.

        Args:
            mlip: The MLIP to be benchmarked.
            output_dir: The output directory of the benchmark. Defaults to
                `benchmark_results`.
        """
        self.mlip = mlip
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: BaseBenchmarkResult = None

    @abstractmethod
    def _generate_data(self) -> Any:
        """Generates any necessary data with `self.mlip`.

        Subclasses must implement this method.

        Returns:
            Raw data from simulations or single-point energy calculations.
                The format can be anything such as a dictionary of structure
                names: simulation trajectories or a list of energy calculations.
        """
        pass

    @abstractmethod
    def _analyze_results(self, raw_data: Any) -> BaseBenchmarkResult:
        """Performs all post-inference or simulation analysis.

        This method processes the raw data generated from
        the generation step to compute final metrics.

        Subclasses must implement this method.

        Args:
            raw_data: Raw data from the data-generation step.

        Returns:
            An instance of `BaseBenchmarkResult`.
        """
        pass


    def run(self) -> BaseBenchmarkResult:
        """Runs the benchmark workflow."""
        raw_data = self._generate_data()
        return self._analyze_results(raw_data)

