import torch.nn as nn

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class StockLogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(StockLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
<<<<<<< Updated upstream
        logits = self.linear(x)
        probabilities = self.sigmoid(logits)
        return probabilities
=======
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

class StockXGBoostModel:
    def __init__(self, input_size, params=None):

        if params is None:
            self.params = {
                'objective': 'binary:logistic',  # For binary classification
                'eval_metric': 'logloss',      # Evaluation metric
                'max_depth': 6,               # Depth of trees
                'learning_rate': 0.1,         # Step size shrinkage
                'n_estimators': 100,          # Number of boosting rounds
                'use_label_encoder': False    # Prevent warnings
            }
        else:
            self.params = params
        
        self.model = None  # Placeholder for the trained model
>>>>>>> Stashed changes
