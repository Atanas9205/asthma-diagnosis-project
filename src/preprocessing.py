import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file and loads it into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw dataset.

    Example
    -------
    >>> df = load_data("data/asthma_disease_data.csv")
    >>> df.shape
    (1000, 28)
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame, missing_threshold: float = 0.4) -> pd.DataFrame:
    """
    Applies a complete data cleaning pipeline:
    - Drops columns with too many missing values
    - Fills numeric NaNs with median
    - Fills categorical NaNs with mode
    - One-hot encodes categorical features
    - Normalizes numerical columns (z-score)

    Parameters
    ----------
    df : pd.DataFrame
        The input raw data to be cleaned and transformed.
    
    missing_threshold : float, optional
        Proportion of missing values above which a column is dropped.
        Default is 0.4 (i.e., 40%).

    Returns
    -------
    pd.DataFrame
        Fully cleaned and numerical dataset, ready for modeling.

    Example
    -------
    >>> raw_df = load_data("data/asthma_disease_data.csv")
    >>> prepared_df = clean_data(raw_df)
    >>> prepared_df.head()
    """
    # Drop columns with too many missing values
    df_cleaned = df.loc[:, df.isnull().mean() < missing_threshold]

    # Detect numeric and categorical columns
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    categorical_cols = df_cleaned.select_dtypes(include='object').columns

    # Fill missing numeric values
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

    # Fill missing categorical values
    df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(df_cleaned[categorical_cols].mode().iloc[0])

    # Encode categorical features
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

    # Normalize numerical features
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    return df_encoded