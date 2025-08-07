import pytest
import pandas as pd
import numpy as np
from src.preprocessing import load_data, clean_data

# Sample mock dataset for testing
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Numeric1': [1, 2, np.nan, 4],
        'Numeric2': [np.nan, 5, 6, 7],
        'Category': ['A', 'B', 'A', None],
        'DropMe': [np.nan, np.nan, np.nan, np.nan]
    })

def test_load_data_reads_csv(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "temp.csv"
    pd.DataFrame({'A': [1, 2]}).to_csv(file_path, index=False)

    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 1)

def test_clean_data_output_shape(sample_dataframe):
    cleaned = clean_data(sample_dataframe, missing_threshold=0.8)
    # DropMe should be removed; Category one-hot encoded; NaNs filled
    assert 'DropMe' not in cleaned.columns
    assert cleaned.isnull().sum().sum() == 0
    assert all(col in cleaned.columns for col in ['Category_B'])  # one-hot encoded

def test_clean_data_scaling(sample_dataframe):
    cleaned = clean_data(sample_dataframe, missing_threshold=0.8)
    # Check if numeric columns are scaled (mean approx. 0, std approx. 1)
    numeric_cols = ['Numeric1', 'Numeric2']
    for col in numeric_cols:
        assert np.isclose(cleaned[col].mean(), 0, atol=1e-7)
        assert np.isclose(cleaned[col].std(ddof=0), 1, atol=1e-7)