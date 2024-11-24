import torch.nn as nn

class StockLogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(StockLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        probabilities = self.sigmoid(logits)
        return probabilities