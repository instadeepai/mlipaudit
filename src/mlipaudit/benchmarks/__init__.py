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
from collections import defaultdict

from mlipaudit.benchmark import Benchmark
from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    BondLengthDistributionBenchmark,
    BondLengthDistributionModelOutput,
    BondLengthDistributionResult,
)
from mlipaudit.benchmarks.conformer_selection.conformer_selection import (
    ConformerSelectionBenchmark,
    ConformerSelectionModelOutput,
    ConformerSelectionResult,
)
from mlipaudit.benchmarks.dihedral_scan.dihedral_scan import (
    DihedralScanBenchmark,
    DihedralScanModelOutput,
    DihedralScanResult,
)
from mlipaudit.benchmarks.folding_stability.folding_stability import (
    FoldingStabilityBenchmark,
    FoldingStabilityModelOutput,
    FoldingStabilityResult,
)
from mlipaudit.benchmarks.noncovalent_interactions.noncovalent_interactions import (
    NoncovalentInteractionsBenchmark,
    NoncovalentInteractionsModelOutput,
    NoncovalentInteractionsResult,
)
from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    NEBModelOutput,
    NEBResult,
    NudgedElasticBandBenchmark,
)
from mlipaudit.benchmarks.reactivity.reactivity import (
    ReactivityBenchmark,
    ReactivityModelOutput,
    ReactivityResult,
)
from mlipaudit.benchmarks.reference_geometry_stability.reference_geometry_stability import (  # noqa: E501
    ReferenceGeometryStabilityBenchmark,
    ReferenceGeometryStabilityModelOutput,
    ReferenceGeometryStabilityResult,
)
from mlipaudit.benchmarks.ring_planarity.ring_planarity import (
    RingPlanarityBenchmark,
    RingPlanarityModelOutput,
    RingPlanarityResult,
)
from mlipaudit.benchmarks.sampling.sampling import (
    SamplingBenchmark,
    SamplingModelOutput,
    SamplingResult,
)
from mlipaudit.benchmarks.scaling.scaling import (
    ScalingBenchmark,
    ScalingModelOutput,
    ScalingResult,
)
from mlipaudit.benchmarks.solvent_radial_distribution.solvent_radial_distribution import (  # noqa: E501
    SolventRadialDistributionBenchmark,
    SolventRadialDistributionModelOutput,
    SolventRadialDistributionResult,
    SolventRadialDistributionStructureResult,
)
from mlipaudit.benchmarks.stability.stability import (
    StabilityBenchmark,
    StabilityModelOutput,
    StabilityResult,
)
from mlipaudit.benchmarks.tautomers.tautomers import (
    TautomersBenchmark,
    TautomersModelOutput,
    TautomersResult,
)
from mlipaudit.benchmarks.water_radial_distribution.water_radial_distribution import (
    WaterRadialDistributionBenchmark,
    WaterRadialDistributionModelOutput,
    WaterRadialDistributionResult,
)

BENCHMARKS = Benchmark.__subclasses__()
BENCHMARK_NAMES = [b.name for b in BENCHMARKS]

BENCHMARKS_WITHOUT_SCORES = [ScalingBenchmark]

# Some benchmarks are still in beta and are not displayed in the public leaderboard
BENCHMARKS_TO_SKIP_FOR_PUBLIC_LEADERBOARD = [NudgedElasticBandBenchmark]


def _setup_benchmark_categories() -> dict[str, list[type[Benchmark]]]:
    mapping = defaultdict(list)
    for b in BENCHMARKS:
        mapping[b.category].append(b)
    return mapping


BENCHMARK_CATEGORIES = _setup_benchmark_categories()

# Dict with keys, values: category, number of benchmarks with scores in the category
BENCHMARK_WITH_SCORES_CATEGORIES = {
    cat: sum(1 for b in benchmarks if b not in BENCHMARKS_WITHOUT_SCORES)
    for cat, benchmarks in BENCHMARK_CATEGORIES.items()
}
BENCHMARK_WITH_SCORES_CATEGORIES_PUBLIC = {
    cat: sum(
        1
        for b in benchmarks
        if b
        not in (BENCHMARKS_WITHOUT_SCORES + BENCHMARKS_TO_SKIP_FOR_PUBLIC_LEADERBOARD)
    )
    for cat, benchmarks in BENCHMARK_CATEGORIES.items()
}
