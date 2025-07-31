from pathlib import Path
from typing import Callable, TypeAlias

import pandas as pd
import streamlit as st
from ase import units

from mlipaudit.dihedral_scan.dihedral_scan import (
    DihedralScanResult,
)

IMG_DIR = Path.cwd() / "app_data" / "small_molecule_conformer_selection" / "img"

ModelName: TypeAlias = str
BenchmarkResultForMultipleModels: TypeAlias = dict[ModelName, DihedralScanResult]


def dihedral_scan_page(
    data_func: Callable[[], BenchmarkResultForMultipleModels],
) -> None:
    """Page for the visualization app for the dihedral scan.

    Args:
        data_func: A data function that delivers the results on request. It does
                   not take any arguments and returns a dictionary with model names as
                   keys and the benchmark results objects as values.
    """
    st.markdown("# Dihedral scan")
    st.sidebar.markdown("# Dihedral scan")

    with st.sidebar.container():
        selected_energy_unit = st.selectbox(
            "Select an energy unit:",
            ["kcal/mol", "eV"],
        )

    st.markdown(
        "Dihedral scans are a common technique in quantum chemistry to study the "
        "effect of bond angles on the energy of a molecule. Here, we assess the "
        "ability of MLIPs to predict the energy profile of a molecule as a function "
        "of the dihedral angle of a rotatable bond in a small molecule."
    )

    st.markdown(
        "We use the TorsionNet 500 test set for this benchmark, which contains 500 "
        "structures of drug-like molecules and their energy profiles around "
        "selected rotatable bonds. The key metric of the benchmark is the average "
        "error of the barrier heights throughtout the dataset, which should be as "
        "small as possible. The barrier height is defined as the maximum of the "
        "energy profile minus the minimum. For further investingation on the model "
        "performance, the deviation of the energy profiles from the reference is "
        "computed based on RMSE and Pearson correlation. A small RMSE and high "
        "Pearson correlation indicates an energy profile that is similar to the "
        "reference."
    )

    st.markdown(
        "For more information, see the [docs](https://mlipaudit-dot-int-research-tpu.uc.r.appspot.com/benchmarks/small-molecules/dihedral_scan.html)."
    )

    # Download data and get model names
    data = data_func()
    # model_names = list(data.keys())
    # model_select = st.sidebar.multiselect(
    #     "Select model(s)", model_names, default=model_names
    # )
    # selected_models = model_select if model_select else model_names

    conversion_factor = (
        1.0 if selected_energy_unit == "kcal/mol" else units.kcal / units.mol
    )
    score_data = [
        {
            "Model name": model_name,
            "MAE": result.avg_mae * conversion_factor,
            "RMSE": result.avg_rmse * conversion_factor,
            "Barrier Height Error": result.avg_barrier_height_error * conversion_factor,
            "Pearson Correlation": result.avg_pearson_r,
        }
        for model_name, result in data.items()
    ]

    # Create dataframe
    df = pd.DataFrame(score_data)
    st.dataframe(df, hide_index=True)
