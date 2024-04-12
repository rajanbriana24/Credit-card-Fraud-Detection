from data_loader import load_data
from data_cleaner import clean_data
from base_model_trainer import train_base_models
from ensemble_trainer import train_stacked_ensemble
from evaluation_metrics import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    """
    Main function to execute the fraud detection pipeline.
    """
    # Load data
    data_path = '/content/creditcard.csv'
    data = load_data(data_path)

    # Clean data
    cleaned_data = clean_data(data)

    # Separate features and target variable
    X = cleaned_data.drop('Class', axis=1)
    y = cleaned_data['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train base models
    predictions = train_base_models(X_train_scaled, X_test_scaled, y_test)

    # Train stacked ensemble model
    auprc, average_precision = train_stacked_ensemble(predictions, y_test)

    # Print results for ensemble model
    print("Stacked Ensemble Model Results:")
    print(f"Area under Precision-Recall Curve (AUPRC): {auprc}")
    print(f"Average Precision Score: {average_precision}")

    # Predictions for ensemble model
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    y_pred = np.where(ensemble_pred > 0.5, 1, 0)

    # Evaluate ensemble model
    accuracy, f1, cm = evaluate_model(y_test, y_pred)
    print("Evaluation Metrics for Ensemble Model:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
