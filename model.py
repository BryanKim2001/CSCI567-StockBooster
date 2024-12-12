import torch.nn as nn

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class StockXGBoostModel:
    def __init__(self, input_size, params=None):

        if params is None:
            self.params = {
                'objective': 'binary:logistic',  # For binary classification
                'eval_metric': 'logloss',      # Evaluation metric
                'max_depth': 6,               # Depth of trees
                'learning_rate': 0.1,         # Step size shrinkage
                'n_estimators': 100,          
                'use_label_encoder': False  
            }
        else:
            self.params = params
        
        self.model = None
