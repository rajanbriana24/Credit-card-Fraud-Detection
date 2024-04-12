import pandas as pd

def load_data(data_path):
    """
    Load the dataset from the specified path.

    Parameters:
    data_path (str): Path to the dataset file.

    Returns:
    pandas.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(data_path)
    return data
