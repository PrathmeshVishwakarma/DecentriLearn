import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms

from averager import federated_averaging
from trainer import train


class net(nn.Module):
    """Model (simple CNN)"""

    def __init__(self):
        super(net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=64)

train_digit_dataset = {}
for digit in range(10):  # for each digit, make seperate dataset
    # Find indices where the label matches the current digit
    indices = [i for i, label in enumerate(train_dataset.targets) if label == digit]

    # Create a specific subset for this digit
    digit_subset = Subset(train_dataset, indices)
    train_digit_dataset[digit] = digit_subset

train_digit_loader = {
    i: DataLoader(dataset=train_digit_dataset[i], shuffle=True, batch_size=32)
    for i in train_digit_dataset.keys()
}


# This will print the same/random digit as its not trained on any number
main_model = net()
device = "cuda" if torch.cuda.is_available() else "cpu"
main_model.to(device)
# dummy_input = torch.randn((32, 1, 28, 28))
# print(torch.argmax(torch.softmax(main_model(dummy_input), dim=1), dim=1))


## To check if model is getting trained, should be around 10% as only 1 digit is trained
# zero = net()
# train(zero, train_digit_loader[0])
# metric = MulticlassAccuracy(num_classes=10)
# for images, labels in test_loader:
#     images, labels = images.to(device), labels.to(device)
#     out = zero(images)
#     print(metric(out, labels) * 100)
#     break


""" Training Phase """
all_models_weights = []

for digit in range(10):
    model = net()
    train(model=model, dataloader=train_digit_loader[digit], device=device)
    all_models_weights.append(model.state_dict())

main_model = federated_averaging(main_model, all_models_weights)
metric = MulticlassAccuracy(num_classes=10)
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    out = main_model(images)
    print(metric(out, labels) * 100)
    break
