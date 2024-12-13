import torch
import numpy as np
import math
import xgboost as xgb

def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].unsqueeze(1).to(device)

            outputs = model(inputs)
            predictions = (outputs > 0.5).float()

            correct += (predictions == labels).sum().item()
            count += labels.size(0)

    accuracy = correct / count
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def evaluate_hybrid_model(hybrid_model, test_loader):
    """
    Evaluates the hybrid model (LSTM + XGBoost) on the test set.
    
    Args:
        hybrid_model: A composite model containing the trained LSTM and XGBoost models.
        test_loader: A DataLoader object containing the test dataset.
    
    Returns:
        accuracy: The accuracy of the hybrid model on the test set.
    """
    lstm_model = hybrid_model.lstm_model
    xgb_model = hybrid_model.xgb_model

    # Ensure LSTM is in evaluation mode
    lstm_model.eval()

    # Collect test data from the DataLoader
    inputs_list = []
    labels_list = []


    # Extract features using the LSTM
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch["label"].cpu().numpy()

            # Extract LSTM features
            if len(inputs.shape) == 2:  # Shape is (batch_size, input_size)
                inputs = inputs.unsqueeze(1) 
            lstm_features = lstm_model(inputs).cpu().numpy()
            inputs_list.append(lstm_features)
            labels_list.append(labels)

    # Stack features and labels
    X_test = np.vstack(inputs_list)
    y_test = np.hstack(labels_list)

    print(f"Extracted Features Shape: {X_test.shape}")
    print(f"Test Labels Shape: {y_test.shape}")

    # Convert to DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)

    y_pred_prob = xgb_model.predict(dtest)


    # Convert probabilities to binary predictions
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

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    return accuracy


