import pandas as pd
from evaluation import evaluate_threshold


def threshold_analysis(probabilities, y_test):

    thresholds = [
        0.10, 0.30,
        0.50, 0.70, 0.90
    ]

    results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            probabilities,
            y_test,
            threshold
        )

        results.append(result)

    return pd.DataFrame(results)