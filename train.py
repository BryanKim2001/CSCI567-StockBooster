import torch
import numpy as np
from torch.utils.data import DataLoader
from model import StockLogisticRegressionModel
from model import StockXGBoostModel
import xgboost as xgb
import logging
logging.basicConfig(level=logging.INFO)

def train_model(train_loader, input_size, num_epochs = 20, learning_rate=0.05, weight_decay=1e-4, device="cpu"):
    model = StockLogisticRegressionModel(input_size).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.6f}")
    return model

def train_xgboost_model(train_loader, num_epochs=200, params=None):
    # Default XGBoost parameters for regression
    if params is None:
        params = {
            "objective": "binary:logistic",  # Binary classification objective
            "eval_metric": "logloss",       # Log loss for binary classification
            "max_depth": 6,
            "learning_rate": 0.05,
            "lambda": 1.0,
            "alpha": 0.0,
            "verbosity": 2 #change back to 1 after debugging
        }

    # Collect all data from the train_loader
    inputs_list = []
    labels_list = []
    for batch in train_loader:
        inputs_list.append(batch["input"].cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())
        #print("hi")

    # Convert to NumPy arrays
    X_train = np.vstack(inputs_list)
    y_train = np.hstack(labels_list)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    print("here")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Data types:", X_train.dtype, y_train.dtype)
    print("DMatrix created:", dtrain.num_row(), "rows,", dtrain.num_col(), "columns")


    # Train the model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_epochs,
        verbose_eval=True
    )
    print("pass")

    return model
