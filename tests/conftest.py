from pathlib import Path

import pytest
from mlip.models import Visnet
from mlip.models.model_io import load_model_from_zip


@pytest.fixture(scope="session")
def load_force_field():
    """Fixture that loads a test force field model from a zip archive.

    Returns:
        The loaded force field.
    """
    model_path = Path(__file__).parent / "model" / "model.zip"
    force_field = load_model_from_zip(Visnet, model_path)
    return force_field


@pytest.fixture(scope="session")
def get_data_input_dir():
    """Fixture to provide a data input directory with test data.

    Returns:
        The path to the test data directory.
    """
    return Path(__file__).parent / "data"
