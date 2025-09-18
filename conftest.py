import pytest
import os
import pandas as pd

@pytest.fixture(scope="session")
def data():
    local_path = os.path.join("", "data", "census.csv")
    df = pd.read_csv(local_path, low_memory=False)
    return df