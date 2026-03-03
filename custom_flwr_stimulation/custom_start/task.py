# Importing modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


# 1. Model Definition: Simple CNN adapted for grayscale MNIST
class net(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super(net, self).__init__()
        # Input channel is 1 for grayscale MNIST
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted for MNIST dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 2. Hard Non-IID Data Partitioning
def load_data(partition_id: int):
    """Load MNIST and return a partition containing only one specific digit."""
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Load full datasets
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Filter indices where label matches the partition_id (0-9)
    # Client with partition_id 0 only gets images of the digit '0', etc.
    train_indices = [
        idx for idx, label in enumerate(trainset.targets) if label == partition_id
    ]
    test_indices = [
        idx for idx, label in enumerate(testset.targets) if label == partition_id
    ]

    # Create DataLoaders for the specific subset
    trainloader = DataLoader(
        Subset(trainset, train_indices), batch_size=32, shuffle=True
    )
    testloader = DataLoader(Subset(testset, test_indices), batch_size=32)

    return trainloader, testloader


def load_test_dataset():
    """Load the complete MNIST test set for global evaluation."""
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
    )
    return test_loader


# 3. Training Logic: Modifies the model in-place
def train_fn(model, trainloader, epochs, device):
    """Train the model on data."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


# 4. Evaluation Logic
def test_fn(net, testloader, device):
    """Validate the network on the provided test set partition."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
