import torch
import numpy as np
from torch.utils.data import DataLoader
from model import StockXGBoostModel
import xgboost as xgb
import logging
logging.basicConfig(level=logging.INFO)

def train_xgboost_model(train_loader, num_epochs=68, params=None):
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",  
            "max_depth": 8,
            "learning_rate": 0.01,
            "lambda": 3.0,
            "alpha": 1.0,
            "verbosity": 2
        }


    inputs_list = []
    labels_list = []
    for batch in train_loader:
        inputs_list.append(batch["input"].cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())

    X_train = np.vstack(inputs_list)
    y_train = np.hstack(labels_list)

    dtrain = xgb.DMatrix(X_train, label=y_train)


    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Data types:", X_train.dtype, y_train.dtype)
    print("DMatrix created:", dtrain.num_row(), "rows,", dtrain.num_col(), "columns")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_epochs,
        verbose_eval=True
    )

    return model
