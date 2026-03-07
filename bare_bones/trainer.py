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

def train_fedprox(model: nn.Module, global_model: nn.Module, dataloader: DataLoader, device="cpu"):
    """Fine tune's with regularisation and returns the weights
    minimize: local_loss(θ_local) + (γ/2) * ||θ_local - θ_global||²
    """
    epochs = 2
    gamma = 0.005
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

def train_pFedMe(model, global_model, dataloader, device, lamda=15, lr=0.01, K=5):
    """
    pFedMe implementation:
    model: The personalized model (w)
    global_model: The global reference (theta)
    lamda: Regularization parameter
    K: Number of personalized update steps
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)   # i dont know why sgd, but okay
    
    # We maintain a reference to global parameters
    theta = list(global_model.parameters())
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Outer loop for pFedMe logic with K
        for _ in range(K):
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # The Moreau Envelope penalty term, same as our regularisation one
            penalty = 0
            for local_param, global_param in zip(model.parameters(), theta):
                penalty += (lamda / 2) * torch.norm(local_param - global_param)**2
            
            (loss + penalty).backward()
            optimizer.step()
            
        # After K steps, we update the global model parameters locally
        # before eventually sending them back to the aggregator
        for local_param, global_param in zip(model.parameters(), theta):
            global_param.data = global_param.data - lr * lamda * (global_param.data - local_param.data)
            
    return model

def test(model: nn.Module, test_loader: DataLoader, device="cpu"):
    model.to(device)
    metric = MulticlassAccuracy(num_classes=10)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        metric.update(out, labels)
    print(metric.compute()*100)
    return metric.compute()*100