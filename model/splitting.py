
import pandas as pd
import numpy as np
import hashlib


def create_sample_split(df, id_column, training_frac=0.8):
    """
    Create sample split based on an ID column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    id_column : str
        The name of the ID column to use for the split.
    training_frac : float, optional
        The fraction of data to be assigned to the training set, by default 0.8.

    Returns
    -------
    pd.DataFrame
        The original dataset with an additional column named 'sample',
        indicating 'train' or 'test' for each row.
    """
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in the DataFrame.")
    
    # Compute the modulo based on the ID column
    if df[id_column].dtype == np.int64:
        # If ID column is numeric, use modulo directly
        modulo = df[id_column] % 100
    else:
        # If ID column is non-numeric, hash the IDs first
        modulo = df[id_column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )
    
    # Assign 'train' or 'test' based on the training fraction
    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")
    
    return df
 

    