import numpy as np

def train_stacked_ensemble(predictions, y_test):
    """
    Train a stacked ensemble model using predictions from base models.

    Parameters:
    predictions (dict): Dictionary containing predictions of base models.
    y_test (numpy.ndarray): Target variable of the testing set.

    Returns:
    tuple: Tuple containing AUPRC and Average Precision Score of the ensemble model.
    """
    # Stack ensemble by averaging predictions
    ensemble_pred = np.mean(list(predictions.values()), axis=0)

    # Calculate precision-recall curve and AUPRC for ensemble model
    precision, recall, _ = precision_recall_curve(y_test, ensemble_pred)
    auprc = auc(recall, precision)
    average_precision = average_precision_score(y_test, ensemble_pred)

    return auprc, average_precision
