import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader



import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)


# Define the LSTM-based Feature Extractor
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMFeatureExtractor, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_size)
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        #batch_size, seq_len, input_size = x.shape
        #x = x.view(-1, input_size)  # Flatten to (batch_size * seq_len, input_size)
        #x = self.bn_input(x)       # Apply BatchNorm
        #x = x.view(batch_size, seq_len, input_size)

        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        self.attention = nn.Linear(64, 1)
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        features = torch.sum(weights * lstm_out, dim=1)
        features = self.bn(features)
        output = self.fc(features)  # (batch_size, output_size)
        return output


def train_hybrid_model(train_loader, num_epochs=68):
    lstm_params = {
        "input_size": 6,  # Number of features (e.g., Open, High, Low, etc.)
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 16,  # Feature size extracted by LSTM
        "learning_rate": 0.001,
        "lstm_epochs": 5,  # Number of epochs to train the LSTM
    }

    # Define XGBoost parameters
    xgboost_params = {
        "objective": "reg:squarederror",  # Regression task
        "eval_metric": "rmse",
        "max_depth": 8,
        "learning_rate": 0.01,
        "lambda": 3.0,
        "alpha": 1.0,
        "verbosity": 2,
    }

    # Step 1: Initialize LSTM feature extractor
    lstm_model = LSTMFeatureExtractor(
        input_size=lstm_params["input_size"],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        output_size=lstm_params["output_size"],
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Loss and optimizer for LSTM (to train LSTM first)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lstm_params["learning_rate"])

    # Step 2: Train the LSTM on the time-series data to extract features
    lstm_model.train()
    for epoch in range(lstm_params["lstm_epochs"]):
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch["input"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch["label"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            #print("Input shape before reshaping:", inputs.shape)

            # Add sequence length dimension if missing
            if len(inputs.shape) == 2:  # Shape is (batch_size, input_size)
                inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, 1, input_size)

            # Debugging input shape after reshaping
            #print("Input shape after reshaping:", inputs.shape)
            optimizer.zero_grad()
            features = lstm_model(inputs)
            loss = criterion(features.squeeze(), labels.float())  # Regression loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch + 1}/{lstm_params['lstm_epochs']}, Loss: {epoch_loss:.4f}")

    # Step 3: Extract features using the trained LSTM
    lstm_model.eval()
    inputs_list = []
    labels_list = []
    with torch.no_grad():
        for batch in train_loader:
            inputs = batch["input"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch["label"].cpu().numpy()
            if len(inputs.shape) == 2:  # Shape is (batch_size, input_size)
                inputs = inputs.unsqueeze(1) 
            features = lstm_model(inputs).cpu().numpy()
            inputs_list.append(features)
            labels_list.append(labels)


    # Convert to NumPy arrays
    X_train = np.vstack(inputs_list)
    y_train = np.hstack(labels_list)


    # Step 4: Train XGBoost on the LSTM-extracted features
    dtrain = xgb.DMatrix(X_train, label=y_train)
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    xgb_model = xgb.train(
        params=xgboost_params,

        dtrain=dtrain,
        num_boost_round=num_epochs,
        verbose_eval=True,
    )
    print("pass")

    return lstm_model, xgb_model


# Define LSTM parameters

# Example usage
# Assuming train_loader is a PyTorch DataLoader containing your dataset
# train_loader = DataLoader(...) 

#lstm_model, xgb_model = train_hybrid_model(train_loader, lstm_params, xgboost_params)
