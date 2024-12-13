import torch.nn as nn
import torch

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class HybridModel:
    def __init__(self, lstm_model, xgb_model):
        """
        Initializes the hybrid model with LSTM and XGBoost components.
        
        Args:
            lstm_model: Trained LSTM model for feature extraction.
            xgb_model: Trained XGBoost model for prediction.
        """
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model

    def predict(self, inputs):
        """
        Makes predictions using the hybrid model.
        
        Args:
            inputs: Torch tensor of inputs to be passed to the LSTM.
        
        Returns:
            numpy array: Predicted outputs (binary or probabilities).
        """
        # Ensure LSTM is in evaluation mode
        self.lstm_model.eval()
        with torch.no_grad():
            # Extract features using the LSTM
            lstm_features = self.lstm_model(inputs).cpu().numpy()

        # Convert to XGBoost DMatrix
        dmatrix = xgb.DMatrix(lstm_features)
        # Use the XGBoost model to make predictions
        y_pred_prob = self.xgb_model.predict(dmatrix)

        return y_pred_prob



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
