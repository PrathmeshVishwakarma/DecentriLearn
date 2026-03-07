import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
import random

from averager import federated_averaging
from trainer import train, test, train_pFedMe


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


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
    ])
target_transform = transforms.Lambda(lambda y: torch.tensor(y))

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform, target_transform=target_transform
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

""" Training the model a little, taking 100 of each digits """
print("Little Training begins: ")
sample_trainer_images, sample_trainer_labels = [], []
for i in train_digit_dataset.keys():
    for _ in range(100):
        idx = random.randint(
            0, 5_000
        )  # select a random index, so we wont be training first 100
        image, label = train_digit_dataset[i][idx]
        sample_trainer_images.append(image)
        sample_trainer_labels.append(label)
sample_trainer_images = torch.stack(sample_trainer_images)
sample_trainer_labels = torch.tensor(sample_trainer_labels)
little_trainer_dataset = TensorDataset(sample_trainer_images, sample_trainer_labels)
little_trainer_dataloader = DataLoader(
    dataset=little_trainer_dataset, shuffle=True, batch_size=32
)

# main_model = train(main_model, little_trainer_dataloader)
# test(main_model, test_loader)

alpha_percent = 10
total_shared_samples = len(little_trainer_dataset)
num_to_sample = int((alpha_percent / 100) * total_shared_samples)

# Access the raw tensors from your balanced "warm-up" set
#shared_images = little_trainer_dataset.tensors[0]
#shared_labels = little_trainer_dataset.tensors[1]
augmented_train_loader = {}

for digit, private_dataset in train_digit_dataset.items():
    indices = random.sample(range(total_shared_samples), num_to_sample)
    shared_subset = Subset(little_trainer_dataset, indices)
    combined_dataset = ConcatDataset([private_dataset, shared_subset])
    augmented_train_loader[digit] = DataLoader(
        dataset=combined_dataset, shuffle=True, batch_size=32
    )


""" Training Phase """
print("Nodal modal training: ")
all_models_weights = []

for digit in range(10):
    local_model = net()
    local_model.load_state_dict(main_model.state_dict())
    local_model = train_pFedMe(model=local_model, global_model=main_model, dataloader=train_digit_loader[digit], device=device)
    all_models_weights.append(local_model.state_dict())

main_model = federated_averaging(main_model, all_models_weights)
test(main_model, test_loader)