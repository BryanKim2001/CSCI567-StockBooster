import torch
from torch.utils.data import DataLoader
from model import StockLogisticRegressionModel

def train_model(train_loader, input_size, num_epochs=50, learning_rate=0.8, weight_decay=1e-4, device="cpu"):
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
