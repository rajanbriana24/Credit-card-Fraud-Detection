def clean_data(data):
    """
    Clean the dataset by removing rows with missing values.

    Parameters:
    data (pandas.DataFrame): The input dataset.

    Returns:
    pandas.DataFrame: The cleaned dataset.
    """
    cleaned_data = data.dropna()  # Drop rows with missing values
    return cleaned_data
