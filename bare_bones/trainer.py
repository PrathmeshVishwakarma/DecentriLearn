import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy


def train(model: nn.Module, dataloader: DataLoader, device="cpu"):
    """Trains and returns the weights"""
    epochs = 2
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    metric = MulticlassAccuracy(num_classes=10)
    model.train()  # set in training mode
    for epoch in range(epochs):
        epoch_loss = 0
        metric.reset()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            metric.update(outputs, labels)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = metric.compute()
        print(f"Epoch {epoch+1} | Loss - {avg_loss} | Accuracy: {accuracy}")
