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

"""NVE energy-conservation benchmark.

Runs short microcanonical (NVE) molecular dynamics trajectories for a few
representative systems and measures how well the model conserves the total
mechanical energy ``E(t) = PE(t) + KE(t)``. The headline metric is the magnitude of
the total-energy drift over the trajectory divided by the standard deviation of the
kinetic energy: a conservative model keeps this dimensionless ratio small, while a
drifting one does not.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from mlip.simulation.enums import MDIntegrator
from pydantic import BaseModel

from mlipaudit.benchmark import (
    Benchmark,
    BenchmarkResult,
    ModelOutput,
)
from mlipaudit.benchmarks.stability.stability import BOX_SIZES
from mlipaudit.benchmarks.water_radial_distribution.water_radial_distribution import (
    WATERBOX_N500,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import ALPHA, compute_metric_score
from mlipaudit.utils import run_simulation, skip_unallowed_elements

logger = logging.getLogger("mlipaudit")

# Cubic box length (Angstrom) of the bulk water system, matching the
# `water_radial_distribution` benchmark's equilibrated 500-molecule box.
WATER_BOX_LENGTH = 24.772

# Scoring threshold on the energy-drift / kinetic-energy-fluctuation ratio
# computed in `_analyze_structure`.
ENERGY_DRIFT_RATIO_THRESHOLD = 1.0

# Both validators of the JAX-MD config must hold: `num_episodes` evenly divides
# `num_steps`, and `snapshot_interval` evenly divides the steps per episode.
SIMULATION_CONFIG: dict[str, Any] = {
    "num_steps": 50_000,
    "num_episodes": 50,
    "snapshot_interval": 50,
    "timestep_fs": 1.0,
    "temperature_kelvin": 300.0,
}
SIMULATION_CONFIG_DEV: dict[str, Any] = {
    "num_steps": 100,
    "num_episodes": 10,
    "snapshot_interval": 10,
    "timestep_fs": 1.0,
    "temperature_kelvin": 300.0,
}

# System classes, used for grouping in the UI / plots.
VACUUM = "vacuum"
BULK_WATER = "bulk_water"
SOLVATED = "solvated"


@dataclass(frozen=True)
class _SystemSpec:
    """Static description of one NVE test system.

    Attributes:
        name: Unique identifier for the system.
        filename: Input structure file, relative to the benchmark data directory.
        system_class: One of ``vacuum``, ``bulk_water`` or ``solvated``.
        box: The periodic box. ``None`` for vacuum, a float for a cubic box, or a
            list of three floats for an orthorhombic box (Angstrom).
        description: Human-readable description of the system.
    """

    name: str
    filename: str
    system_class: str
    box: float | list[float] | None
    description: str


SYSTEMS: list[_SystemSpec] = [
    _SystemSpec(
        name="Small_molecule_HCNO",
        filename="small_molecule_HCNO.xyz",
        system_class=VACUUM,
        box=None,
        description="Small molecule (HCNO) in vacuum",
    ),
    _SystemSpec(
        name="Water_box_n500",
        filename=WATERBOX_N500,
        system_class=BULK_WATER,
        box=WATER_BOX_LENGTH,
        description="Bulk water (500 molecules)",
    ),
    _SystemSpec(
        name="Peptide_solvated",
        filename="peptide_solv.xyz",
        system_class=SOLVATED,
        box=BOX_SIZES["Peptide_solvated"],
        description="Solvated Oxytocin (PDB: 7OFG)",
    ),
    _SystemSpec(
        name="Peptide_solvated_ions",
        filename="peptide_solv_ion.xyz",
        system_class=SOLVATED,
        box=BOX_SIZES["Peptide_solvated_ions"],
        description="Solvated Neurotensin with counter-ions (PDB: 2LNF)",
    ),
]

SYSTEMS_BY_NAME: dict[str, _SystemSpec] = {spec.name: spec for spec in SYSTEMS}


class NVEStructureResult(BaseModel):
    """Per-system result for the NVE energy-conservation benchmark.

    Attributes:
        structure_name: The system identifier.
        description: Human-readable description of the system.
        system_class: One of ``vacuum``, ``bulk_water`` or ``solvated``.
        num_atoms: The number of atoms in the system.
        num_frames: The number of (finite) trajectory frames used for the fit.
        times_ps: The snapshot times in picoseconds (x-axis of the drift plot).
        energy_drift_ev: The total-energy drift in eV at each snapshot.
        drift_slope_ev_per_ps: Slope of the linear fit to the drift (eV / ps).
        total_drift_ev: Magnitude of the fitted total-energy drift over the whole
            trajectory (eV), i.e. ``|slope| * duration``.
        kinetic_energy_std_ev: Standard deviation of the kinetic energy along the
            trajectory (eV); the fluctuation scale used to normalize the drift.
        energy_drift_ratio: ``total_drift_ev / kinetic_energy_std_ev``, the
            dimensionless energy-conservation metric used for scoring.
        intercept_ev: Intercept of the linear fit (eV).
        skipped: Whether the system was skipped because the model does not support
            one of its elements.
        failed: Whether the simulation ran but failed (e.g. it blew up).
        score: The per-system score between 0 and 1, or None if skipped.
    """

    structure_name: str
    description: str
    system_class: str
    num_atoms: int = 0
    num_frames: int = 0
    times_ps: list[float] = []
    energy_drift_ev: list[float] = []
    drift_slope_ev_per_ps: float | None = None
    total_drift_ev: float | None = None
    kinetic_energy_std_ev: float | None = None
    energy_drift_ratio: float | None = None
    intercept_ev: float | None = None
    skipped: bool = False
    failed: bool = False
    score: float | None = None


class NVEEnergyConservationResult(BenchmarkResult):
    """Result object for the NVE energy-conservation benchmark.

    Attributes:
        structure_results: The list of per-system results.
        n_skipped_unallowed_elements: How many systems were skipped because the
            model does not support one of their elements.
        failed: Whether all systems were skipped or failed.
        score: The overall benchmark score between 0 and 1.
    """

    structure_results: list[NVEStructureResult]
    n_skipped_unallowed_elements: int = 0


class NVEEnergyConservationModelOutput(ModelOutput):
    """Raw model outputs for the NVE energy-conservation benchmark.

    Per-system entries are aligned by index with ``structure_names``. Entries are
    ``None`` for systems that were skipped or whose simulation failed.

    Attributes:
        structure_names: The ordered list of system identifiers that were run.
        num_atoms: The number of atoms per system.
        times_ps: The snapshot times in picoseconds per system.
        potential_energies_ev: The per-snapshot potential energies (eV) per system.
        kinetic_energies_ev: The per-snapshot kinetic energies (eV) per system.
        skipped_structures: Systems skipped due to unsupported elements.
        n_skipped_unallowed_elements: Number of skipped systems.
    """

    structure_names: list[str]
    num_atoms: list[int | None]
    times_ps: list[list[float] | None]
    potential_energies_ev: list[list[float] | None]
    kinetic_energies_ev: list[list[float] | None]
    skipped_structures: list[str]
    n_skipped_unallowed_elements: int = 0


class NVEEnergyConservationBenchmark(Benchmark):
    """Benchmark measuring total-energy conservation under NVE dynamics.

    For each test system, a short NVE (velocity-Verlet, no thermostat) trajectory
    is run with velocities initialized from a Maxwell-Boltzmann distribution. The
    total mechanical energy ``PE + KE`` is tracked over the trajectory and its drift
    relative to the first frame is fitted with a linear model; the fitted drift over
    the run, divided by the standard deviation of the kinetic energy, is mapped to a
    score.

    Attributes:
        name: The unique benchmark name, ``nve_energy_conservation``.
        category: The UI grouping category, ``General``.
        result_class: The `NVEEnergyConservationResult` type.
        model_output_class: The `NVEEnergyConservationModelOutput` type.
        required_elements: The union of elements across all test systems. Used as
            metadata only; per-system element gating is done in `run_model` (see
            `skip_if_elements_missing`).
        skip_if_elements_missing: Set to False so the benchmark is never skipped
            wholesale; instead, individual systems whose elements the model cannot
            handle are skipped within `run_model`.
    """

    name = "nve_energy_conservation"
    category = "General"
    result_class = NVEEnergyConservationResult
    model_output_class = NVEEnergyConservationModelOutput

    required_elements = {"H", "C", "N", "O", "S", "Na", "Cl"}
    skip_if_elements_missing = False

    def run_model(self) -> None:
        """Run an NVE trajectory and record total energy for each system.

        Systems whose elements the model does not support are skipped. For the
        rest, an NVE MD simulation is run and the per-snapshot potential and
        kinetic energies are stored in `model_output`.
        """
        systems = self._systems

        skipped = set(
            skip_unallowed_elements(
                self.force_field,
                [
                    (name, list(atoms.get_chemical_symbols()))
                    for name, atoms in systems.items()
                ],
            )
        )
        if skipped:
            logger.info(
                "Skipping %d systems with unallowed elements: %s", len(skipped), skipped
            )

        output = NVEEnergyConservationModelOutput(
            structure_names=[],
            num_atoms=[],
            times_ps=[],
            potential_energies_ev=[],
            kinetic_energies_ev=[],
            skipped_structures=sorted(skipped),
            n_skipped_unallowed_elements=len(skipped),
        )

        for name, atoms in systems.items():
            spec = SYSTEMS_BY_NAME[name]
            output.structure_names.append(name)
            output.num_atoms.append(len(atoms))

            if name in skipped:
                output.times_ps.append(None)
                output.potential_energies_ev.append(None)
                output.kinetic_energies_ev.append(None)
                continue

            logger.info("Running NVE MD for %s (%d atoms)", name, len(atoms))
            kwargs = dict(self._md_kwargs)
            if spec.box is not None:
                kwargs["box"] = spec.box
            state = run_simulation(
                atoms,
                self.force_field,
                md_integrator=MDIntegrator.NVE_VELOCITY_VERLET,
                **kwargs,
            )

            if (
                state is None
                or state.kinetic_energy is None
                or state.potential_energy is None
            ):
                logger.info("Simulation failed for %s", name)
                output.times_ps.append(None)
                output.potential_energies_ev.append(None)
                output.kinetic_energies_ev.append(None)
                continue

            kinetic_energies = np.asarray(state.kinetic_energy, dtype=float)
            potential_energies = np.asarray(state.potential_energy, dtype=float)
            num_frames = min(len(kinetic_energies), len(potential_energies))
            times_ps = (
                np.arange(num_frames)
                * self._md_kwargs["snapshot_interval"]
                * self._md_kwargs["timestep_fs"]
                / 1000.0
            )

            output.times_ps.append(times_ps.tolist())
            output.potential_energies_ev.append(
                potential_energies[:num_frames].tolist()
            )
            output.kinetic_energies_ev.append(kinetic_energies[:num_frames].tolist())

        self.model_output = output

    def analyze(self) -> NVEEnergyConservationResult:
        """Compute the drift, fit its slope and score each system.

        Returns:
            An `NVEEnergyConservationResult` with per-system drift curves, fitted
            slopes and scores, plus the aggregate benchmark score.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        output = self.model_output
        structure_results: list[NVEStructureResult] = []

        for index, name in enumerate(output.structure_names):
            spec = SYSTEMS_BY_NAME[name]
            num_atoms = output.num_atoms[index] or 0

            if name in output.skipped_structures:
                structure_results.append(
                    NVEStructureResult(
                        structure_name=name,
                        description=spec.description,
                        system_class=spec.system_class,
                        num_atoms=num_atoms,
                        skipped=True,
                    )
                )
                continue

            times = output.times_ps[index]
            potential = output.potential_energies_ev[index]
            kinetic = output.kinetic_energies_ev[index]

            result = self._analyze_structure(spec, num_atoms, times, potential, kinetic)
            structure_results.append(result)

        scored = [
            r.score for r in structure_results if not r.skipped and r.score is not None
        ]
        if not scored:
            return NVEEnergyConservationResult(
                structure_results=structure_results,
                n_skipped_unallowed_elements=output.n_skipped_unallowed_elements,
                failed=True,
            )

        return NVEEnergyConservationResult(
            structure_results=structure_results,
            n_skipped_unallowed_elements=output.n_skipped_unallowed_elements,
            score=float(np.mean(scored)),
        )

    def _analyze_structure(
        self,
        spec: _SystemSpec,
        num_atoms: int,
        times: list[float] | None,
        potential: list[float] | None,
        kinetic: list[float] | None,
    ) -> NVEStructureResult:
        """Compute the drift fit and score for a single system.

        Args:
            spec: The system specification.
            num_atoms: The number of atoms in the system.
            times: The snapshot times in picoseconds, or None if skipped/failed.
            potential: The per-snapshot potential energies (eV), or None.
            kinetic: The per-snapshot kinetic energies (eV), or None.

        Returns:
            The per-system result with the drift curve, fitted slope and score.
        """
        base = {
            "structure_name": spec.name,
            "description": spec.description,
            "system_class": spec.system_class,
            "num_atoms": num_atoms,
        }

        if times is None or potential is None or kinetic is None:
            return NVEStructureResult(**base, failed=True, score=0.0)

        time_array = np.asarray(times, dtype=float)
        total = np.asarray(potential, dtype=float) + np.asarray(kinetic, dtype=float)

        # A valid NVE trajectory has a finite energy at every snapshot; any non-finite
        # value means the run diverged, so treat it as a failure.
        # A linear fit also needs at least two points.
        if total.size < 2 or not np.all(np.isfinite(total)):
            return NVEStructureResult(**base, failed=True, score=0.0)

        drift = total - total[0]
        slope, intercept = np.polyfit(time_array, drift, 1)

        kinetic_energy_std_ev = float(np.std(np.asarray(kinetic, dtype=float)))
        if kinetic_energy_std_ev == 0.0:
            # No kinetic-energy fluctuation means this is not a valid MD trajectory,
            # and there is no fluctuation scale to normalize the drift against.
            return NVEStructureResult(**base, failed=True, score=0.0)

        # Conservation metric: fitted total-energy drift over the run, normalized by
        # the kinetic-energy fluctuation scale.
        duration_ps = float(time_array[-1] - time_array[0])
        total_drift_ev = abs(float(slope) * duration_ps)
        energy_drift_ratio = total_drift_ev / kinetic_energy_std_ev
        score = float(
            compute_metric_score(
                np.array([energy_drift_ratio]), ENERGY_DRIFT_RATIO_THRESHOLD, ALPHA
            )[0]
        )

        return NVEStructureResult(
            **base,
            num_frames=int(total.size),
            times_ps=time_array.tolist(),
            energy_drift_ev=drift.tolist(),
            drift_slope_ev_per_ps=float(slope),
            total_drift_ev=total_drift_ev,
            kinetic_energy_std_ev=kinetic_energy_std_ev,
            energy_drift_ratio=energy_drift_ratio,
            intercept_ev=float(intercept),
            score=score,
        )

    @functools.cached_property
    def _systems(self) -> dict[str, Atoms]:
        """Load the run-mode-selected test systems as ASE atoms (charge/spin set)."""
        systems: dict[str, Atoms] = {}
        for spec in self._selected_specs:
            atoms = ase_read(self.data_input_dir / self.name / spec.filename)
            systems[spec.name] = atoms
        return systems

    @property
    def _selected_specs(self) -> list[_SystemSpec]:
        """The systems to run for the current run mode (dev runs only vacuum)."""
        if self.run_mode == RunMode.DEV:
            return [spec for spec in SYSTEMS if spec.system_class == VACUUM]
        return SYSTEMS

    @functools.cached_property
    def _md_kwargs(self) -> dict[str, Any]:
        """The simulation configuration for the current run mode."""
        if self.run_mode == RunMode.DEV:
            return SIMULATION_CONFIG_DEV
        return SIMULATION_CONFIG
