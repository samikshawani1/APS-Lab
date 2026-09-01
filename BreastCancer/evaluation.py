from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_threshold(probabilities, y_test, threshold):

    actual_malignant = (y_test.values == 0).astype(int)

    predicted_malignant = (
        probabilities[:, 0] >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual_malignant,
        predicted_malignant
    ).ravel()

    return {
        "Threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy_score(
            actual_malignant,
            predicted_malignant
        ),
        "Precision": precision_score(
            actual_malignant,
            predicted_malignant,
            zero_division=0
        ),
        "Recall": recall_score(
            actual_malignant,
            predicted_malignant,
            zero_division=0
        ),
        "F1 Score": f1_score(
            actual_malignant,
            predicted_malignant,
            zero_division=0
        )
    }