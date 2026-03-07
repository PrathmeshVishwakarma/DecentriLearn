import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import math


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

    return model

def train_with_regularisation(model: nn.Module, global_model: nn.Module, dataloader: DataLoader, device="cpu"):
    """Fine tune's with regularisation and returns the weights
    minimize: local_loss(θ_local) + (γ/2) * ||θ_local - θ_global||²
    """
    epochs = 2
    gamma = 0.01
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
            local_loss = criterion(outputs, labels)

            proximal_loss = 0.0
            for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                proximal_loss += (local_param - global_param).pow(2).sum()

            total_loss = local_loss + (gamma / 2) * proximal_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            metric.update(outputs, labels)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = metric.compute()
        print(f"Epoch {epoch+1} | Loss - {avg_loss} | Accuracy: {accuracy}")

    return model


def test(model: nn.Module, test_loader: DataLoader, device="cpu"):
    model.to(device)
    metric = MulticlassAccuracy(num_classes=10)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        metric.update(out, labels)
    print(metric.compute()*100)