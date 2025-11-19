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

import enum


class RunMode(enum.Enum):
    """Enum for the mode of a benchmark run.

    Attributes:
        DEV: Very minimal and fast. Just meant for testing.
        FAST: For some long-running benchmarks, a limited set of test cases is run to
              decrease overall runtime. For most benchmarks, this is not different
              from the standard case.
        STANDARD: Complete run of all benchmark cases.

    """

    DEV = "dev"
    FAST = "fast"
    STANDARD = "standard"
