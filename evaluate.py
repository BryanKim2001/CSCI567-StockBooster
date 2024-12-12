import torch
import numpy as np
import xgboost as xgb
import math

def evaluate_xgboost_model(model, test_loader):
    correct = 0
    count = 0

    inputs_list = []
    labels_list = []
    print(len(test_loader))
    for batch in test_loader:
        inputs_list.append(batch["input"].cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())

    X_test = np.vstack(inputs_list)
    y_test = np.hstack(labels_list)
    print(X_test.shape, y_test.shape)
    print(len(X_test), len(y_test))

    dtest = xgb.DMatrix(X_test)
    y_pred_prob = model.predict(dtest)

    y_pred = (y_pred_prob > 0.5).astype(int)

    correct = np.sum(y_pred == y_test)
    count = len(y_test)

    accuracy = correct / count

    true_positive = np.sum((y_pred == 1) & (y_test == 1))
    true_negative = np.sum((y_pred == 0) & (y_test == 0))
    false_positive = np.sum((y_pred == 1) & (y_test == 0))
    false_negative = np.sum((y_pred == 0) & (y_test == 1))

    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = math.sqrt(
        (true_positive + false_positive) * (true_positive + false_negative) *
        (true_negative + false_positive) * (true_negative + false_negative)
    )
    mcc = numerator / denominator if denominator != 0 else 0

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    return accuracy

