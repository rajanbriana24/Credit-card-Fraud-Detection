import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.knn import KNN
from pyod.models.abod import ABOD

def train_base_models(X_train_scaled, X_test_scaled, y_test):
    """
    Train various anomaly detection models and evaluate their performance.

    Parameters:
    X_train_scaled (numpy.ndarray): Scaled features of the training set.
    X_test_scaled (numpy.ndarray): Scaled features of the testing set.
    y_test (numpy.ndarray): Target variable of the testing set.

    Returns:
    dict: Dictionary containing predictions of base models.
    """
    # Define base models
    base_models = {
        'Isolation Forest': IsolationForest(n_estimators=40, contamination=0.1, random_state=42),
        'One-Class SVM': OneClassSVM(gamma='auto'),
        'Local Outlier Factor': LocalOutlierFactor(novelty=True, contamination=0.1),
        'Autoencoder': AutoEncoder(hidden_neurons=[32, 16, 16, 32], epochs=20, batch_size=64, contamination=0.1),
        'Gaussian Mixture Models': GaussianMixture(n_components=2, covariance_type='full', random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'K-Nearest Neighbors': KNN(contamination=0.1),
        'Minimum Covariance Determinant': EllipticEnvelope(contamination=0.1),
        'Angle-based Outlier Detection': ABOD()
    }

    # Train base models and make predictions
    predictions = {}
    for model_name, model in base_models.items():
        print(f"Training and predicting with {model_name}...")
        model.fit(X_train_scaled)
        if hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test_scaled)
        elif hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_score = model.predict(X_test_scaled)
        predictions[model_name] = y_score

    return predictions
