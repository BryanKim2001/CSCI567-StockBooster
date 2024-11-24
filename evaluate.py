import torch

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