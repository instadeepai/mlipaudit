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
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.calculators.calculator import all_changes

from mlipaudit.benchmarks import DihedralScanBenchmark, FoldingStabilityBenchmark

INPUT_DATA_DIR = Path(__file__).parent / "data"


class DummyASECalculator(ASECalculator):
    """Dummy calculator for testing."""

    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(self):
        """Overridden constructor."""
        self.allowed_atomic_numbers = {
            "H",
            "C",
            "N",
            "O",
            "S",
            "P",
            "F",
            "Cl",
        }
        ASECalculator.__init__(self)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Assigns zero energy and forces."""
        ASECalculator.calculate(self, atoms, properties, system_changes)
        if "energy" in properties:  # type: ignore
            self.results["energy"] = np.array(0.0)
        if "forces" in properties:  # type: ignore
            self.results["forces"] = np.zeros_like(atoms.get_positions())  # type: ignore


@pytest.mark.parametrize(
    "benchmark_class", [DihedralScanBenchmark, FoldingStabilityBenchmark]
)
def test_benchmarks_can_be_run_with_ase_calculator(benchmark_class, mocker):
    """Tests for two benchmarks (one using inference, one using simulation), whether
    they can also be run with an ASE calculator as the force field.
    """
    if benchmark_class is DihedralScanBenchmark:
        # Needs to be mocked because the limited test data will produce NaNs
        _mocked_spearman_r = mocker.patch(
            "mlipaudit.benchmarks.dihedral_scan.dihedral_scan.pearsonr",
            side_effect=lambda x, y: (1.0, 0.0),
        )

    force_field = DummyASECalculator()
    benchmark = benchmark_class(force_field, INPUT_DATA_DIR, "dev")

    benchmark.run_model()

    # Check that model outputs are indeed zero
    if benchmark_class is DihedralScanBenchmark:
        for fragment_output in benchmark.model_output.fragments:
            assert not any(fragment_output.energy_predictions)
    else:
        sim_state = benchmark.model_output.simulation_states[0]
        assert not np.any(sim_state.forces)

    results = benchmark.analyze()

    # If no validation errors until now, the produces values must be reasonable,
    # however, what we know is that with zero energies and forces, we should get a
    # low score for dihedral scan, but a decent one for folding stability.
    if benchmark_class is DihedralScanBenchmark:
        assert results.score < 0.1
    else:
        assert results.score > 0.3
