from sklearn.model_selection import train_test_split

from data import load_data
from model import train_model
from predictions import get_predictions
from threshold_analysis import threshold_analysis


def main():

    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    # Get probabilities
    probabilities, results = get_predictions(
        model,
        X_test,
        y_test
    )

    # Display predictions
    print("\nPrediction Results:")
    print(results.head(10))

    # Threshold analysis
    table = threshold_analysis(
        probabilities,
        y_test
    )

    print("\nThreshold Analysis:")
    print(table)


if __name__ == "__main__":
    main()