import pandas as pd


def get_predictions(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)

    results = pd.DataFrame({
        "Actual_Class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1]
    })

    return probabilities, results