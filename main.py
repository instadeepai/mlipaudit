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
"""Main script to run a full benchmark."""

from huggingface_hub import hf_hub_download
from mlip.models import Visnet
from mlip.models.model_io import load_model_from_zip

from mlipaudit.small_molecule_conformer_selection import ConformerSelectionBenchmark


def main():
    """Main for the MLIPAudit benchmark."""
    hf_hub_download(
        repo_id="InstaDeepAI/visnet-organics",
        filename="visnet_organics_01.zip",
        local_dir="models/",
    )

    force_field = load_model_from_zip(Visnet, "models/visnet_organics_01.zip")

    benchmarks = [ConformerSelectionBenchmark(force_field)]

    print(benchmarks)


if __name__ == "__main__":
    main()
