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
from typing import Callable, TypeAlias, get_args, get_origin

import pytest
from streamlit.testing.v1 import AppTest

from mlipaudit.benchmark import Benchmark, BenchmarkResult
from mlipaudit.reactivity import ReactivityBenchmark
from mlipaudit.ring_planarity import RingPlanarityBenchmark
from mlipaudit.ui import reactivity_page, ring_planarity_page

BenchmarkResultForMultipleModels: TypeAlias = dict[str, BenchmarkResult]


def _get_data_func_for_benchmark(
    benchmark_class: type[Benchmark],
) -> Callable[[], BenchmarkResultForMultipleModels]:
    def data_func() -> BenchmarkResultForMultipleModels:
        kwargs_for_result = {}
        for name, field in benchmark_class.result_class.model_fields.items():
            if field.annotation is float:
                kwargs_for_result[name] = 0.675
                continue

            annotation_origin = get_origin(field.annotation)
            if annotation_origin in [list, dict]:
                idx = 0 if annotation_origin is list else 1
                subresult_class = get_args(field.annotation)[idx]
                kwargs_for_subresult = {}
                for subname, subfield in subresult_class.model_fields.items():
                    if subfield.annotation is float:
                        kwargs_for_subresult[subname] = 0.4
                    if subfield.annotation is list[float]:
                        kwargs_for_subresult[subname] = [0.3, 0.5]  # type: ignore
                    if subfield.annotation is str:
                        kwargs_for_subresult[subname] = "test"  # type: ignore

                if annotation_origin is list:
                    kwargs_for_result[name] = [subresult_class(**kwargs_for_subresult)]  # type: ignore
                else:
                    kwargs_for_result[name] = {
                        "test": subresult_class(**kwargs_for_subresult)  # type: ignore
                    }

        return {
            "model_1": benchmark_class.result_class(**kwargs_for_result),
            "model_2": benchmark_class.result_class(**kwargs_for_result),
        }

    return data_func


def _app_script(page_func, data_func):
    import functools  # noqa

    import streamlit as st  # noqa

    page = st.Page(
        functools.partial(
            page_func,
            data_func=data_func,
        ),
        title="Benchmark",
        url_path="benchmark",
    )

    pages_to_show = [page]
    pg = st.navigation(pages_to_show)
    pg.run()


@pytest.mark.parametrize(
    "benchmark_to_test, page_to_test",
    [
        (RingPlanarityBenchmark, ring_planarity_page),
        (ReactivityBenchmark, reactivity_page),
    ],
)
def test_ui_page_is_working_correctly(benchmark_to_test, page_to_test):
    """Tests a UI page with dummy data and the AppTest pattern from streamlit."""
    dummy_data_func = _get_data_func_for_benchmark(benchmark_to_test)

    print(dummy_data_func())

    args_for_app = (page_to_test, dummy_data_func)
    app = AppTest.from_function(_app_script, args=args_for_app)

    app.run()
    assert not app.exception
