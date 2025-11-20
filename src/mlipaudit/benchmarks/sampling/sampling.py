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
from collections import defaultdict

import numpy as np
from ase.io import read as ase_read
from mdtraj.core.topology import Residue
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.benchmarks.sampling.helpers import (
    calculate_distribution_hellinger_distance,
    calculate_distribution_rmsd,
    calculate_multidimensional_distribution,
    get_all_dihedrals_from_trajectory,
    identify_outlier_data_points,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import (
    create_mdtraj_trajectory_from_simulation_state,
    run_simulation,
)
from mlipaudit.utils.simulation import REUSABLE_BIOMOLECULES_OUTPUTS_ID
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")

STRUCTURE_NAMES = [
    "chignolin_1uao_xray",
    "trp_cage_2jof_xray",
    "orexin_beta_1cq0_nmr",
]

BOX_SIZES = {
    "chignolin_1uao_xray": [23.98, 22.45, 20.68],
    "trp_cage_2jof_xray": [29.33, 29.74, 23.59],
    "orexin_beta_1cq0_nmr": [40.30, 29.56, 33.97],
}

SIMULATION_CONFIG = {
    "num_steps": 250_000,
    "snapshot_interval": 10_000,
    "num_episodes": 25,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}
NUM_DEV_SYSTEMS = 1
NUM_FAST_SYSTEMS = 2

RESNAME_TO_BACKBONE_RESIDUE_TYPE = {
    "GLY": "GLY",
    "ILE": "ILE_VAL",
    "VAL": "ILE_VAL",
    "PRO": "PRO",
}

SIDECHAIN_DIHEDRAL_COUNTS = {
    # 0 chi angles
    "GLY": 0,
    "ALA": 0,
    # 1 chi angle
    "SER": 1,
    "THR": 1,
    "CYS": 1,
    "PRO": 1,
    "VAL": 1,
    # 2 chi angles
    "ASN": 2,
    "ASP": 2,
    "HIS": 2,
    "ILE": 2,
    "LEU": 2,
    "PHE": 2,
    "TYR": 2,
    "TRP": 2,
    # 3 chi angles
    "GLU": 3,
    "GLN": 3,
    "MET": 3,
    # 4 chi angles
    "ARG": 4,
    "LYS": 4,
}

OUTLIERS_RATIO_BACKBONE_SCORE_THRESHOLD = 0.1
OUTLIERS_RATIO_SIDECHAIN_SCORE_THRESHOLD = 0.03


class ResidueTypeBackbone(BaseModel):
    """Stores reference backbone dihedral data for a residue type.

    Attributes:
        phi: The reference phi dihedral values for the residue type.
        psi: The reference psi dihedral values for the residue type.
    """

    phi: list[float]
    psi: list[float]


class ResidueTypeSidechain(BaseModel):
    """Stores reference sidechain dihedral data for a residue type.

    Attributes:
        chi1: The reference chi1 dihedral values for the residue type.
        chi2: The reference chi2 dihedral values for the residue type.
        chi3: The reference chi3 dihedral values for the residue type.
        chi4: The reference chi4 dihedral values for the residue type.
        chi5: The reference chi5 dihedral values for the residue type.
    """

    chi1: list[float] | None = None
    chi2: list[float] | None = None
    chi3: list[float] | None = None
    chi4: list[float] | None = None
    chi5: list[float] | None = None


ReferenceDataBackbone = TypeAdapter(dict[str, ResidueTypeBackbone])
ReferenceDataSidechain = TypeAdapter(dict[str, ResidueTypeSidechain])


class SamplingSystemResult(BaseModel):
    """Stores the result for one system of the sampling benchmark.

    Attributes:
        structure_name: The name of the structure.
        rmsd_backbone_dihedrals: The RMSD of the backbone dihedral distribution
            with respect to the reference data for each residue type.
        hellinger_distance_backbone_dihedrals: The Hellinger distance of the backbone
            dihedral distribution with respect to the reference data for each residue
            type.
        rmsd_sidechain_dihedrals: The RMSD of the sidechain dihedral distribution
            with respect to the reference data for each residue type.
        hellinger_distance_sidechain_dihedrals: The Hellinger distance of the sidechain
            dihedral distribution with respect to the reference data for each residue
            type.
        outliers_ratio_backbone_dihedrals: The ratio of outliers in the backbone
            dihedral distribution for each residue type.
        outliers_ratio_sidechain_dihedrals: The ratio of outliers in the sidechain
            dihedral distribution for each residue type.
        failed: Whether the simulation was stable. If not stable, the other
            attributes will be not be set.
    """

    structure_name: str

    rmsd_backbone_dihedrals: dict[str, float] | None = None
    hellinger_distance_backbone_dihedrals: dict[str, float] | None = None
    rmsd_sidechain_dihedrals: dict[str, float] | None = None
    outliers_ratio_backbone_dihedrals: dict[str, float] | None = None
    hellinger_distance_sidechain_dihedrals: dict[str, float] | None = None
    outliers_ratio_sidechain_dihedrals: dict[str, float] | None = None

    failed: bool = False


class SamplingResult(BenchmarkResult):
    """Stores the result of the sampling benchmark.

    Attributes:
        systems: The result for each system, including those that failed.
        exploded_systems: The systems that exploded, or that failed
            during simulation.
        rmsd_backbone_total: The RMSD of the backbone dihedral distribution
            for all systems.
        hellinger_distance_backbone_total: The Hellinger distance of the backbone
            dihedral distribution for all systems.
        rmsd_sidechain_total: The RMSD of the sidechain dihedral distribution
            for all systems.
        hellinger_distance_sidechain_total: The Hellinger distance of the sidechain
            dihedral distribution for all systems.
        outliers_ratio_backbone_total: The ratio of outliers in the backbone
            dihedral distribution for all systems.
        outliers_ratio_sidechain_total: The ratio of outliers in the sidechain
            dihedral distribution for all systems.
        rmsd_backbone_dihedrals: The RMSD of the backbone dihedral distribution
            for each residue type.
        hellinger_distance_backbone_dihedrals: The Hellinger distance of the backbone
            dihedral distribution for each residue type.
        rmsd_sidechain_dihedrals: The RMSD of the sidechain dihedral distribution
            for each residue type.
        hellinger_distance_sidechain_dihedrals: The Hellinger distance of the sidechain
            dihedral distribution for each residue type.
        outliers_ratio_backbone_dihedrals: The ratio of outliers in the backbone
            dihedral distribution for each residue type.
        outliers_ratio_sidechain_dihedrals: The ratio of outliers in the sidechain
            dihedral distribution for each residue type.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    systems: list[SamplingSystemResult]

    exploded_systems: list[str]

    rmsd_backbone_total: float | None = None
    hellinger_distance_backbone_total: float | None = None
    rmsd_sidechain_total: float | None = None
    hellinger_distance_sidechain_total: float | None = None

    outliers_ratio_backbone_total: float | None = None
    outliers_ratio_sidechain_total: float | None = None

    rmsd_backbone_dihedrals: dict[str, float] | None = None
    hellinger_distance_backbone_dihedrals: dict[str, float] | None = None
    rmsd_sidechain_dihedrals: dict[str, float] | None = None
    hellinger_distance_sidechain_dihedrals: dict[str, float] | None = None
    outliers_ratio_backbone_dihedrals: dict[str, float] | None = None
    outliers_ratio_sidechain_dihedrals: dict[str, float] | None = None


class SamplingModelOutput(ModelOutput):
    """Stores model outputs for the sampling benchmark.

    Attributes:
        structure_names: The names of the structures.
        simulation_states: `SimulationState` or `None` object for
            each structure in the same order as the structure
            names. `None` if the simulation failed.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SamplingBenchmark(Benchmark):
    """Benchmark for sampling of amino acid backbone and sidechain dihedrals.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `sampling`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Biomolecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is `SamplingResult`.
        model_output_class: A reference to the `SamplingModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
        reusable_output_id: An optional ID that references other benchmarks with
            identical input systems and `ModelOutput` signatures (in form of a tuple).
            If present, a user or the CLI can make use of this information to reuse
            cached model outputs from another benchmark carrying the same ID instead of
            rerunning simulations or inference.
    """

    name = "sampling"
    category = "Biomolecules"
    result_class = SamplingResult
    model_output_class = SamplingModelOutput

    required_elements = {"N", "H", "O", "S", "C"}

    reusable_output_id = REUSABLE_BIOMOLECULES_OUTPUTS_ID

    def run_model(self) -> None:
        """Run an MD simulation for each system."""
        if self.run_mode == RunMode.DEV:
            structure_names = STRUCTURE_NAMES[:NUM_DEV_SYSTEMS]
        elif self.run_mode == RunMode.FAST:
            structure_names = STRUCTURE_NAMES[:NUM_FAST_SYSTEMS]
        else:
            structure_names = STRUCTURE_NAMES

        if self.run_mode == RunMode.DEV:
            md_kwargs = SIMULATION_CONFIG_DEV
        else:
            md_kwargs = SIMULATION_CONFIG

        self.model_output = SamplingModelOutput(
            structure_names=[],
            simulation_states=[],
        )

        for structure_name in structure_names:
            logger.info("Running MD for %s", structure_name)

            xyz_filename = structure_name + ".xyz"
            atoms = ase_read(
                self.data_input_dir / self.name / "starting_structures" / xyz_filename
            )

            simulation_state = run_simulation(
                atoms, self.force_field, box=BOX_SIZES[structure_name], **md_kwargs
            )

            self.model_output.structure_names.append(structure_name)
            self.model_output.simulation_states.append(simulation_state)

    def analyze(self) -> SamplingResult:
        """Analyze the sampling benchmark.

        Raises:
            RuntimeError: If `run_model()` has not been called first.

        Returns:
            The result of the sampling benchmark.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        self._assert_structure_names_in_model_output()

        # Reference data preparation

        backbone_reference_data, sidechain_reference_data = self._reference_data()
        reference_backbone_dihedral_distributions = self._get_reference_distributions(
            backbone_reference_data
        )
        reference_sidechain_dihedral_distributions = self._get_reference_distributions(
            sidechain_reference_data
        )

        histograms_reference_backbone_dihedrals = {}
        histograms_reference_sidechain_dihedrals = {}

        for (
            residue_name,
            array_of_dihedrals,
        ) in reference_backbone_dihedral_distributions.items():
            hist, _ = calculate_multidimensional_distribution(array_of_dihedrals)
            histograms_reference_backbone_dihedrals[residue_name] = hist

        for (
            residue_name,
            array_of_dihedrals,
        ) in reference_sidechain_dihedral_distributions.items():
            hist, _ = calculate_multidimensional_distribution(array_of_dihedrals)
            histograms_reference_sidechain_dihedrals[residue_name] = hist

        # End of reference data preparation

        systems = []
        failed_systems = []
        num_stable = 0

        for i, structure_name in enumerate(self.model_output.structure_names):
            simulation_state = self.model_output.simulation_states[i]

            if simulation_state is None or not is_simulation_stable(simulation_state):
                molecule_result = SamplingSystemResult(
                    structure_name=structure_name, failed=True
                )
                systems.append(molecule_result)
                failed_systems.append(structure_name)
                continue

            num_stable += 1
            box_size = BOX_SIZES[structure_name]

            trajectory = create_mdtraj_trajectory_from_simulation_state(
                simulation_state,
                topology_path=(
                    self.data_input_dir
                    / self.name
                    / "pdb_reference_structures"
                    / f"{structure_name}.pdb"
                ),
                cell_lengths=box_size,  # type: ignore
            )

            dihedrals_data = get_all_dihedrals_from_trajectory(trajectory)

            distribution_metrics = self._analyze_distribution(
                dihedrals_data,
                histograms_reference_backbone_dihedrals,
                histograms_reference_sidechain_dihedrals,
            )

            outlier_metrics = self._analyze_outliers(
                dihedrals_data,
                reference_backbone_dihedral_distributions,
                reference_sidechain_dihedral_distributions,
            )

            systems.append(
                SamplingSystemResult(
                    structure_name=structure_name,
                    rmsd_backbone_dihedrals=distribution_metrics["rmsd_backbone"],
                    hellinger_distance_backbone_dihedrals=distribution_metrics[
                        "hellinger_distance_backbone"
                    ],
                    rmsd_sidechain_dihedrals=distribution_metrics["rmsd_sidechain"],
                    hellinger_distance_sidechain_dihedrals=distribution_metrics[
                        "hellinger_distance_sidechain"
                    ],
                    outliers_ratio_backbone_dihedrals=outlier_metrics[
                        "outliers_ratio_backbone_dihedrals"
                    ],
                    outliers_ratio_sidechain_dihedrals=outlier_metrics[
                        "outliers_ratio_sidechain_dihedrals"
                    ],
                )
            )
        if num_stable == 0:
            return SamplingResult(
                systems=systems, exploded_systems=failed_systems, score=0.0
            )

        avg_rmsd_backbone = self._average_metrics_per_residue(
            systems,
            "rmsd_backbone_dihedrals",
        )
        avg_hellinger_distance_backbone = self._average_metrics_per_residue(
            systems,
            "hellinger_distance_backbone_dihedrals",
        )
        avg_rmsd_sidechain = self._average_metrics_per_residue(
            systems,
            "rmsd_sidechain_dihedrals",
        )
        avg_hellinger_distance_sidechain = self._average_metrics_per_residue(
            systems,
            "hellinger_distance_sidechain_dihedrals",
        )

        avg_outliers_ratio_backbone = self._average_metrics_per_residue(
            systems,
            "outliers_ratio_backbone_dihedrals",
        )
        avg_outliers_ratio_sidechain = self._average_metrics_per_residue(
            systems,
            "outliers_ratio_sidechain_dihedrals",
        )

        score = compute_benchmark_score(
            [
                list(avg_outliers_ratio_backbone.values()),
                list(avg_outliers_ratio_sidechain.values()),
            ],
            [
                OUTLIERS_RATIO_BACKBONE_SCORE_THRESHOLD,
                OUTLIERS_RATIO_SIDECHAIN_SCORE_THRESHOLD,
            ],
        )

        return SamplingResult(
            systems=systems,
            exploded_systems=failed_systems,
            rmsd_backbone_dihedrals=avg_rmsd_backbone,
            hellinger_distance_backbone_dihedrals=avg_hellinger_distance_backbone,
            rmsd_sidechain_dihedrals=avg_rmsd_sidechain,
            hellinger_distance_sidechain_dihedrals=avg_hellinger_distance_sidechain,
            outliers_ratio_backbone_dihedrals=avg_outliers_ratio_backbone,
            outliers_ratio_sidechain_dihedrals=avg_outliers_ratio_sidechain,
            outliers_ratio_backbone_total=self._average_over_residues(
                avg_outliers_ratio_backbone
            ),
            outliers_ratio_sidechain_total=self._average_over_residues(
                avg_outliers_ratio_sidechain
            ),
            rmsd_backbone_total=self._average_over_residues(avg_rmsd_backbone),
            hellinger_distance_backbone_total=self._average_over_residues(
                avg_hellinger_distance_backbone
            ),
            rmsd_sidechain_total=self._average_over_residues(avg_rmsd_sidechain),
            hellinger_distance_sidechain_total=self._average_over_residues(
                avg_hellinger_distance_sidechain
            ),
            score=score,
        )

    def _analyze_distribution(
        self,
        dihedrals_data: dict[Residue, dict[str, np.ndarray]],
        reference_backbone_dihedral_distributions: dict[str, np.ndarray],
        reference_sidechain_dihedral_distributions: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Analyze the distribution of dihedrals.

        Args:
            dihedrals_data: The dihedral data from the simulation.
            reference_backbone_dihedral_distributions: The reference distributions for
                the backbone dihedrals.
            reference_sidechain_dihedral_distributions: The reference distributions for
                the sidechain dihedrals.

        Returns:
            The distribution metrics for the dihedrals.
        """
        distribution_metrics: dict[str, dict[str, float]] = {
            "rmsd_backbone": {},
            "rmsd_sidechain": {},
            "hellinger_distance_backbone": {},
            "hellinger_distance_sidechain": {},
        }

        unique_residue_names = set([residue.name for residue in dihedrals_data.keys()])

        sampled_backbone_dihedral_distributions = self._get_sampled_distributions(
            dihedrals_data,
            backbone=True,
        )
        sampled_sidechain_dihedral_distributions = self._get_sampled_distributions(
            dihedrals_data,
            backbone=False,
        )

        for residue_name in unique_residue_names:
            reference_backbone_residue_type = RESNAME_TO_BACKBONE_RESIDUE_TYPE.get(
                residue_name, "GENERAL"
            )

            hist_sampled_backbone, _ = calculate_multidimensional_distribution(
                sampled_backbone_dihedral_distributions[residue_name]
            )

            rmsd_backbone = calculate_distribution_rmsd(
                reference_backbone_dihedral_distributions[
                    reference_backbone_residue_type
                ],
                hist_sampled_backbone,
            )

            hellinger_distance_backbone = calculate_distribution_hellinger_distance(
                reference_backbone_dihedral_distributions[
                    reference_backbone_residue_type
                ],
                hist_sampled_backbone,
            )

            distribution_metrics["rmsd_backbone"][residue_name] = rmsd_backbone
            distribution_metrics["hellinger_distance_backbone"][residue_name] = (
                hellinger_distance_backbone
            )

        for residue_name in unique_residue_names:
            if residue_name in sampled_sidechain_dihedral_distributions:
                hist_sampled_sidechain, _ = calculate_multidimensional_distribution(
                    sampled_sidechain_dihedral_distributions[residue_name]
                )

                rmsd_sidechain = calculate_distribution_rmsd(
                    reference_sidechain_dihedral_distributions[residue_name],
                    hist_sampled_sidechain,
                )

                hellinger_distance_sidechain = (
                    calculate_distribution_hellinger_distance(
                        reference_sidechain_dihedral_distributions[residue_name],
                        hist_sampled_sidechain,
                    )
                )

                distribution_metrics["rmsd_sidechain"][residue_name] = rmsd_sidechain
                distribution_metrics["hellinger_distance_sidechain"][residue_name] = (
                    hellinger_distance_sidechain
                )

        return distribution_metrics

    def _analyze_outliers(
        self,
        dihedrals_data: dict[Residue, dict[str, np.ndarray]],
        reference_backbone_dihedral_distributions: dict[str, np.ndarray],
        reference_sidechain_dihedral_distributions: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Analyze the outliers in the sampled dihedral distributions.

        Args:
            dihedrals_data: The dihedral data from the simulation.
            reference_backbone_dihedral_distributions: The reference backbone dihedral
                distributions.
            reference_sidechain_dihedral_distributions: The reference sidechain dihedral
                distributions.

        Returns:
            The outlier metrics.
        """
        sampled_backbone_dihedral_distributions = self._get_sampled_distributions(
            dihedrals_data,
            backbone=True,
        )
        sampled_sidechain_dihedral_distributions = self._get_sampled_distributions(
            dihedrals_data,
            backbone=False,
        )

        outlier_metrics: dict[str, dict[str, float]] = {
            "outliers_ratio_backbone_dihedrals": {},
            "outliers_ratio_sidechain_dihedrals": {},
        }

        for (
            residue_name,
            array_of_dihedrals,
        ) in sampled_backbone_dihedral_distributions.items():
            reference_backbone_res_type = RESNAME_TO_BACKBONE_RESIDUE_TYPE.get(
                residue_name, "GENERAL"
            )

            outliers_backbone = identify_outlier_data_points(
                array_of_dihedrals,
                reference_backbone_dihedral_distributions[reference_backbone_res_type],
            )
            outliers_ratio_backbone = np.sum(outliers_backbone) / len(outliers_backbone)
            outlier_metrics["outliers_ratio_backbone_dihedrals"][residue_name] = (
                outliers_ratio_backbone
            )

        for (
            residue_name,
            array_of_dihedrals,
        ) in sampled_sidechain_dihedral_distributions.items():
            outliers_sidechain = identify_outlier_data_points(
                array_of_dihedrals,
                reference_sidechain_dihedral_distributions[residue_name],
            )
            outliers_ratio_sidechain = np.sum(outliers_sidechain) / len(
                outliers_sidechain
            )
            outlier_metrics["outliers_ratio_sidechain_dihedrals"][residue_name] = (
                outliers_ratio_sidechain
            )

        return outlier_metrics

    def _reference_data(
        self,
    ) -> tuple[dict[str, ResidueTypeBackbone], dict[str, ResidueTypeSidechain]]:
        with open(
            self.data_input_dir / self.name / "backbone_reference_data.json",
            "r",
            encoding="utf-8",
        ) as f:
            backbone_reference_data = ReferenceDataBackbone.validate_json(f.read())
        with open(
            self.data_input_dir / self.name / "sidechain_reference_data.json",
            "r",
            encoding="utf-8",
        ) as f:
            sidechain_reference_data = ReferenceDataSidechain.validate_json(f.read())

        return backbone_reference_data, sidechain_reference_data

    def _get_reference_distributions(
        self,
        reference_dihedrals: dict[str, ResidueTypeBackbone]
        | dict[str, ResidueTypeSidechain],
    ) -> dict[str, np.ndarray]:
        """Get the reference distributions for the dihedrals.

        Args:
            reference_dihedrals: The reference dihedrals data for all residue types.

        Returns:
            The reference distributions column-stacked into a single array.
        """
        reference_distributions: dict[str, np.ndarray] = {}

        unique_residue_names = set(reference_dihedrals.keys())
        if isinstance(next(iter(reference_dihedrals.values())), ResidueTypeBackbone):
            backbone = True
        else:
            backbone = False

        for residue_name in unique_residue_names:
            if backbone:
                dihedral_keys = ["phi", "psi"]
            else:
                dihedral_keys = self._get_allowed_sidechain_dihedral_keys(residue_name)
                if len(dihedral_keys) == 0:
                    continue

            reference_distributions[residue_name] = np.column_stack([
                getattr(reference_dihedrals[residue_name], dihedral_key)
                for dihedral_key in dihedral_keys
            ])

        return reference_distributions

    def _get_sampled_distributions(
        self,
        dihedrals_data: dict[Residue, dict[str, np.ndarray]],
        backbone: bool = True,
    ) -> dict[str, np.ndarray]:
        """Get the sampled dihedral distributions.

        Args:
            dihedrals_data: The dihedral data from the simulation.
            backbone: Whether to get the backbone dihedral distributions. If False,
                the sidechain dihedral distributions will be returned.

        Returns:
            The sampled dihedral distributions column-stacked into a single array.
        """
        sampled_distributions: dict[str, np.ndarray] = {}

        if backbone:
            dihedral_keys = ["phi", "psi"]

        unique_residue_names = set([residue.name for residue in dihedrals_data.keys()])

        dihedrals_per_unique_name: dict[str, dict[str, np.ndarray]] = {}
        for residue, dihedrals in dihedrals_data.items():
            if residue.name not in dihedrals_per_unique_name:
                dihedrals_per_unique_name[residue.name] = defaultdict(list)
            for dihedral_type, angle_list in dihedrals.items():
                dihedrals_per_unique_name[residue.name][dihedral_type].extend(
                    angle_list
                )

        for residue_name in unique_residue_names:
            if not backbone:
                dihedral_keys = self._get_allowed_sidechain_dihedral_keys(residue_name)
                if len(dihedral_keys) == 0:
                    continue

            sampled_distributions[residue_name] = np.column_stack([
                dihedrals_per_unique_name[residue_name][dihedral_key]
                for dihedral_key in dihedral_keys
            ])

        return sampled_distributions

    def _get_allowed_sidechain_dihedral_keys(
        self,
        residue_name: str,
    ) -> list[str]:
        """Get the allowed sidechain dihedral keys for a residue type.

        Args:
            residue_name: The name of the residue type.

        Returns:
            The allowed sidechain dihedral keys for the residue type.
        """
        if SIDECHAIN_DIHEDRAL_COUNTS[residue_name] == 0:
            return []

        return [f"chi{i + 1}" for i in range(SIDECHAIN_DIHEDRAL_COUNTS[residue_name])]

    def _average_metrics_per_residue(
        self,
        metrics_per_system: list[SamplingSystemResult],
        metric_name: str,
    ) -> dict[str, float]:
        """Average the distribution metrics across all systems
        that were stable.

        Args:
            metrics_per_system: The metrics per system.
            metric_name: The name of the metric to average.

        Returns:
            The average metrics per residue.
        """
        average_metrics: dict[str, float] = {}
        metric_per_residue: dict[str, list[float]] = defaultdict(list)

        stable_systems = [s for s in metrics_per_system if not s.failed]

        for system in stable_systems:
            system_metrics = getattr(system, metric_name)
            for residue_name, metric in system_metrics.items():
                metric_per_residue[residue_name].append(metric)

        for residue_name, metrics in metric_per_residue.items():
            average_metrics[residue_name] = np.mean(metrics)

        return average_metrics

    def _average_over_residues(
        self,
        metrics_per_residue: dict[str, float],
    ) -> float:
        """Average the distribution metrics across all residues.

        Args:
            metrics_per_residue: The metrics per residue.

        Returns:
            The average metrics.
        """
        return np.mean(list(metrics_per_residue.values()))

    def _assert_structure_names_in_model_output(self) -> None:
        """Asserts whether model output structure names are correct as they may
        have been transferred from a different benchmark.
        """
        assert set(self.model_output.structure_names).issubset(STRUCTURE_NAMES)  # type: ignore
        assert len(self.model_output.structure_names) == (  # type: ignore
            NUM_DEV_SYSTEMS
            if self.run_mode == RunMode.DEV
            else (
                NUM_FAST_SYSTEMS
                if self.run_mode == RunMode.FAST
                else len(STRUCTURE_NAMES)
            )
        )
