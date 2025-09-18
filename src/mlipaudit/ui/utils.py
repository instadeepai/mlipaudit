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

import pandas as pd
import streamlit as st

DEFAULT_IMAGE_DOWNLOAD_PPI = 300


def _color_score_blue_gradient(val):
    normalized_val = max(0, min(1, val))
    blue_intensity = int(normalized_val * 255)
    return f"background-color: rgb(255, 255, {blue_intensity})"


def display_model_scores(df: pd.DataFrame) -> None:
    """Display model scores in a table."""
    df_sorted = df[["Model name", "Score"]].sort_values(by="Score", ascending=False)
    st.dataframe(
        df_sorted.style.format(precision=3),
        hide_index=True,
    )


@st.cache_resource
def create_st_image(image_path: Path, caption: str | None = None) -> st.image:
    """Image creation helper that is cached.

    Args:
        image_path: Path to image.
        caption: Caption string. Can be None, which is the default.

    Returns:
        The streamlit image object.
    """
    return st.image(image_path, caption=caption)
