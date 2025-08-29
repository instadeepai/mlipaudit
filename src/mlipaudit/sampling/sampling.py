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
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.sampling.helpers import (
    calculate_distribution_kl_divergence,
    calculate_distribution_rmsd,
    calculate_multidimensional_distribution,
    get_all_dihedrals_from_trajectory,
    identify_outlier_data_points,
)
from mlipaudit.utils import create_mdtraj_trajectory_from_simulation_state

logger = logging.getLogger("mlipaudit")

STRUCTURE_NAMES = [
    "ala_leu_glu_lys_sol",
    "gln_arg_asp_ala_sol",
    "glu_gly_ser_arg_sol",
    "gly_thr_trp_gly_sol",
    "gly_tyr_ala_val_sol",
    "met_ser_asn_gly_sol",
    "met_val_his_asn_sol",
    "pro_met_ile_gln_sol",
    "pro_met_phe_ala_sol",
    "ser_ala_cys_pro_sol",
    "trp_phe_gly_ala_sol",
    "val_glu_lys_ala_sol",
]

CUBIC_BOX_SIZES = {
    "ala_leu_glu_lys_sol": 20.7,
    "gln_arg_asp_ala_sol": 20.7,
    "glu_gly_ser_arg_sol": 20.7,
    "gly_thr_trp_gly_sol": 21.1,
    "gly_tyr_ala_val_sol": 20.7,
    "met_ser_asn_gly_sol": 20.9,
    "met_val_his_asn_sol": 21.0,
    "pro_met_ile_gln_sol": 20.8,
    "pro_met_phe_ala_sol": 20.9,
    "ser_ala_cys_pro_sol": 20.8,
    "trp_phe_gly_ala_sol": 20.8,
    "val_glu_lys_ala_sol": 20.8,
}

SIMULATION_CONFIG = {
    "num_steps": 150_000,
    "snapshot_interval": 1000,
    "num_episodes": 150,
    "temperature_kelvin": 350.0,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 1,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 350.0,
}

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


class ResidueTypeBackbone(BaseModel):
    """Stores reference backbone dihedral data for a residue type.

    Attributes:
        name: The name of the residue type.
        phi: The reference phi dihedral values for the residue type.
        psi: The reference psi dihedral values for the residue type.
    """

    phi: list[float]
    psi: list[float]


class ResidueTypeSidechain(BaseModel):
    """Stores reference sidechain dihedral data for a residue type.

    Attributes:
        name: The name of the residue type.
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
        kl_divergence_backbone_dihedrals: The KL divergence of the backbone dihedral
            distribution with respect to the reference data for each residue type.
        rmsd_sidechain_dihedrals: The RMSD of the sidechain dihedral distribution
            with respect to the reference data for each residue type.
        kl_divergence_sidechain_dihedrals: The KL divergence of the sidechain
            dihedral distribution with respect to the reference data for each residue
            type.
        outliers_ratio_backbone_dihedrals: The ratio of outliers in the backbone
            dihedral distribution for each residue type.
        outliers_ratio_sidechain_dihedrals: The ratio of outliers in the sidechain
            dihedral distribution for each residue type.
        outliers_ratio_backbone_total: The ratio of outliers in the backbone
            dihedral distribution for all residue types.
        outliers_ratio_sidechain_total: The ratio of outliers in the sidechain
            dihedral distribution for all residue types.
    """

    structure_name: str

    rmsd_backbone_dihedrals: dict[str, float]
    kl_divergence_backbone_dihedrals: dict[str, float]
    rmsd_sidechain_dihedrals: dict[str, float]
    kl_divergence_sidechain_dihedrals: dict[str, float]
    outliers_ratio_backbone_dihedrals: dict[str, float]
    outliers_ratio_sidechain_dihedrals: dict[str, float]


class SamplingResult(BenchmarkResult):
    """Stores the result of the sampling benchmark."""

    systems: list[SamplingSystemResult]

    exploded_systems: list[str]

    rmsd_backbone_total: float
    kl_divergence_backbone_total: float
    rmsd_sidechain_total: float
    kl_divergence_sidechain_total: float

    outliers_ratio_backbone_total: float
    outliers_ratio_sidechain_total: float

    rmsd_backbone_dihedrals: dict[str, float]
    kl_divergence_backbone_dihedrals: dict[str, float]
    rmsd_sidechain_dihedrals: dict[str, float]
    kl_divergence_sidechain_dihedrals: dict[str, float]
    outliers_ratio_backbone_dihedrals: dict[str, float]
    outliers_ratio_sidechain_dihedrals: dict[str, float]


class SamplingModelOutput(ModelOutput):
    """Stores model outputs for the sampling benchmark."""

    structure_names: list[str]
    simulation_states: list[SimulationState]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SamplingBenchmark(Benchmark):
    """Benchmark for sampling of amino acid backbone and sidechain dihedrals.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `sampling`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is `SamplingResult`.
    """

    name = "sampling"
    result_class = SamplingResult

    def run_model(self) -> None:
        """Run an MD simulation for each system."""
        self.model_output = SamplingModelOutput(
            structure_names=[],
            simulation_states=[],
        )

        if self.fast_dev_run:
            md_config_dict = SIMULATION_CONFIG_FAST
            structure_names = ["ala_leu_glu_lys_sol"]
        else:
            md_config_dict = SIMULATION_CONFIG
            structure_names = STRUCTURE_NAMES

        for structure_name in structure_names:
            logger.info("Running MD for %s", structure_name)
            xyz_filename = structure_name + ".xyz"
            box_size = CUBIC_BOX_SIZES[structure_name]
            md_config = JaxMDSimulationEngine.Config(
                **md_config_dict,
                box=box_size,
            )
            atoms = ase_read(
                self.data_input_dir / self.name / "starting_structures" / xyz_filename
            )

            md_engine = JaxMDSimulationEngine(atoms, self.force_field, md_config)
            md_engine.run()

            final_state = md_engine.state
            self.model_output.structure_names.append(structure_name)
            self.model_output.simulation_states.append(final_state)

    def analyze(self) -> SamplingResult:
        """Analyze the sampling benchmark.

        Raises:
            RuntimeError: If `run_model()` has not been called first.

        Returns:
            The result of the sampling benchmark.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        systems = []
        skipped_systems = []

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

        for i, structure_name in enumerate(self.model_output.structure_names):
            simulation_state = self.model_output.simulation_states[i]
            box_size = CUBIC_BOX_SIZES[structure_name]

            trajectory = create_mdtraj_trajectory_from_simulation_state(
                simulation_state,
                topology_path=(
                    self.data_input_dir
                    / self.name
                    / "pdb_reference_structures"
                    / f"{structure_name}.pdb"
                ),
                cell_lengths=(box_size / 10.0, box_size / 10.0, box_size / 10.0),
            )

            dihedrals_data = get_all_dihedrals_from_trajectory(trajectory)

            skip = False
            for residue_dict in dihedrals_data.values():
                for arr in residue_dict.values():
                    if np.isnan(arr).any():
                        skip = True
                        break
                if skip:
                    break
            if skip:
                logger.warning(
                    "Skipping system %s because it contains nan values. "
                    "This likely means the system has exploded.",
                    structure_name,
                )
                skipped_systems.append(structure_name)
                continue

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
                    kl_divergence_backbone_dihedrals=distribution_metrics[
                        "kl_divergence_backbone"
                    ],
                    rmsd_sidechain_dihedrals=distribution_metrics["rmsd_sidechain"],
                    kl_divergence_sidechain_dihedrals=distribution_metrics[
                        "kl_divergence_sidechain"
                    ],
                    outliers_ratio_backbone_dihedrals=outlier_metrics[
                        "outliers_ratio_backbone_dihedrals"
                    ],
                    outliers_ratio_sidechain_dihedrals=outlier_metrics[
                        "outliers_ratio_sidechain_dihedrals"
                    ],
                )
            )

        avg_rmsd_backbone = self._average_metrics(
            systems,
            "rmsd_backbone_dihedrals",
        )
        avg_kl_divergence_backbone = self._average_metrics(
            systems,
            "kl_divergence_backbone_dihedrals",
        )
        avg_rmsd_sidechain = self._average_metrics(
            systems,
            "rmsd_sidechain_dihedrals",
        )
        avg_kl_divergence_sidechain = self._average_metrics(
            systems,
            "kl_divergence_sidechain_dihedrals",
        )

        avg_outliers_ratio_backbone = self._average_metrics(
            systems,
            "outliers_ratio_backbone_dihedrals",
        )
        avg_outliers_ratio_sidechain = self._average_metrics(
            systems,
            "outliers_ratio_sidechain_dihedrals",
        )

        return SamplingResult(
            systems=systems,
            exploded_systems=skipped_systems,
            rmsd_backbone_dihedrals=avg_rmsd_backbone,
            kl_divergence_backbone_dihedrals=avg_kl_divergence_backbone,
            rmsd_sidechain_dihedrals=avg_rmsd_sidechain,
            kl_divergence_sidechain_dihedrals=avg_kl_divergence_sidechain,
            outliers_ratio_backbone_dihedrals=avg_outliers_ratio_backbone,
            outliers_ratio_sidechain_dihedrals=avg_outliers_ratio_sidechain,
            outliers_ratio_backbone_total=self._average_over_residues(
                avg_outliers_ratio_backbone
            ),
            outliers_ratio_sidechain_total=self._average_over_residues(
                avg_outliers_ratio_sidechain
            ),
            rmsd_backbone_total=self._average_over_residues(avg_rmsd_backbone),
            kl_divergence_backbone_total=self._average_over_residues(
                avg_kl_divergence_backbone
            ),
            rmsd_sidechain_total=self._average_over_residues(avg_rmsd_sidechain),
            kl_divergence_sidechain_total=self._average_over_residues(
                avg_kl_divergence_sidechain
            ),
        )

    def _analyze_distribution(
        self,
        dihedrals_data: dict[Residue, dict[str, np.ndarray]],
        histograms_reference_backbone_dihedrals: dict[str, np.ndarray],
        histograms_reference_sidechain_dihedrals: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Analyze the distribution of dihedrals.

        Args:
            dihedrals_data: The dihedral data from the simulation.
            histograms_reference_backbone_dihedrals: The reference distributions for
                the backbone dihedrals.
            histograms_reference_sidechain_dihedrals: The reference distributions for
                the sidechain dihedrals.

        Returns:
            The distribution metrics for the dihedrals.
        """
        distribution_metrics: dict[str, dict[str, float]] = {
            "rmsd_backbone": {},
            "rmsd_sidechain": {},
            "kl_divergence_backbone": {},
            "kl_divergence_sidechain": {},
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
            reference_backbone_residue_type = self._get_backbone_reference_key(
                residue_name
            )

            hist_sampled_backbone, _ = calculate_multidimensional_distribution(
                sampled_backbone_dihedral_distributions[residue_name]
            )

            rmsd_backbone = calculate_distribution_rmsd(
                hist_sampled_backbone,
                histograms_reference_backbone_dihedrals[
                    reference_backbone_residue_type
                ],
            )

            kl_divergence_backbone = calculate_distribution_kl_divergence(
                histograms_reference_backbone_dihedrals[
                    reference_backbone_residue_type
                ],
                hist_sampled_backbone,
            )

            distribution_metrics["rmsd_backbone"][residue_name] = rmsd_backbone
            distribution_metrics["kl_divergence_backbone"][residue_name] = (
                kl_divergence_backbone
            )

        for residue_name in unique_residue_names:
            if residue_name in sampled_sidechain_dihedral_distributions:
                hist_sampled_sidechain, _ = calculate_multidimensional_distribution(
                    sampled_sidechain_dihedral_distributions[residue_name]
                )

                rmsd_sidechain = calculate_distribution_rmsd(
                    hist_sampled_sidechain,
                    histograms_reference_sidechain_dihedrals[residue_name],
                )

                kl_divergence_sidechain = calculate_distribution_kl_divergence(
                    histograms_reference_sidechain_dihedrals[residue_name],
                    hist_sampled_sidechain,
                )

                distribution_metrics["rmsd_sidechain"][residue_name] = rmsd_sidechain
                distribution_metrics["kl_divergence_sidechain"][residue_name] = (
                    kl_divergence_sidechain
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
            reference_backbone_res_type = self._get_backbone_reference_key(residue_name)

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

        for residue_name in unique_residue_names:
            if not backbone:
                dihedral_keys = self._get_allowed_sidechain_dihedral_keys(residue_name)
                if len(dihedral_keys) == 0:
                    continue

            sampled_distributions[residue_name] = np.column_stack([
                dihedrals_data[residue][dihedral_key]
                for residue in dihedrals_data.keys()
                for dihedral_key in dihedral_keys
                if residue.name == residue_name
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

    def _average_metrics(
        self,
        metrics_per_system: list[SamplingSystemResult],
        metric_name: str,
    ) -> dict[str, float]:
        """Average the distribution metrics across all systems.

        Args:
            metrics_per_system: The metrics per system.
            metric_name: The name of the metric to average.

        Returns:
            The average metrics per residue.
        """
        average_metrics: dict[str, float] = {}
        metric_per_residue: dict[str, list[float]] = defaultdict(list)

        for system in metrics_per_system:
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

    def _get_backbone_reference_key(
        self,
        residue_name: str,
    ) -> str:
        """Get the reference key for the backbone dihedral distributions.

        Args:
            residue_name: The name of the residue type.

        Returns:
            The reference key for the backbone dihedral distributions.
        """
        if residue_name in RESNAME_TO_BACKBONE_RESIDUE_TYPE:
            return RESNAME_TO_BACKBONE_RESIDUE_TYPE[residue_name]
        else:
            return "GENERAL"
