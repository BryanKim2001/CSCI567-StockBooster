import torch
import numpy as np
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

def evaluate_xgboost_model(model, test_loader):
    correct = 0
    count = 0

    # Collect test data from the DataLoader
    inputs_list = []
    labels_list = []
    print(len(test_loader))
    for batch in test_loader:
        inputs_list.append(batch["input"].cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())

    # Convert to NumPy arrays
    X_test = np.vstack(inputs_list)
    y_test = np.hstack(labels_list)
    print(X_test.shape, y_test.shape)
    print(len(X_test), len(y_test))

    dtest = xgb.DMatrix(X_test)
    print("here now")
    y_pred_prob = model.predict(dtest)

    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate accuracy
    correct = np.sum(y_pred == y_test)
    count = len(y_test)

    accuracy = correct / count
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

