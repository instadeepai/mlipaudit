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
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

from mlipaudit.benchmark import Benchmark, BenchmarkResult

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, BenchmarkResult]


class UIPageWrapper(ABC):
    """Wrapper around the UI page functions. Allows to simplify code in app.py."""

    @classmethod
    @abstractmethod
    def get_page_func(
        cls,
    ) -> Callable[[Callable[[], BenchmarkResultForMultipleModels]], None]:
        """Returns the page function implementation for the given UI page.

        Returns:
            The page function.
        """
        pass

    @classmethod
    @abstractmethod
    def get_benchmark_class(cls) -> type[Benchmark]:
        """Returns the associated benchmark class for the given UI page.

        Returns:
            The benchmark class.
        """
        pass
