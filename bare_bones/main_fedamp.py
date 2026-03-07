import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
import random

from averager import federated_averaging, fedamp_averaging
from trainer import train, test, train_with_regularisation

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

test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64)

train_digit_dataset = {}
for digit in range(10):  
    indices = [i for i, label in enumerate(train_dataset.targets) if label == digit]
    digit_subset = Subset(train_dataset, indices)
    train_digit_dataset[digit] = digit_subset

# ---------------------------------------------------------
# Warm-up Phase
# ---------------------------------------------------------
main_model = net()
device = "cuda" if torch.cuda.is_available() else "cpu"
main_model.to(device)

print("\n--- Little Training (Warm-up) Begins ---")
sample_trainer_images, sample_trainer_labels = [], []
for i in train_digit_dataset.keys():
    for _ in range(100):
        idx = random.randint(0, 5_000)  
        image, label = train_digit_dataset[i][idx]
        sample_trainer_images.append(image)
        sample_trainer_labels.append(label)

sample_trainer_images = torch.stack(sample_trainer_images)
sample_trainer_labels = torch.tensor(sample_trainer_labels)
little_trainer_dataset = TensorDataset(sample_trainer_images, sample_trainer_labels)
little_trainer_dataloader = DataLoader(dataset=little_trainer_dataset, shuffle=True, batch_size=32)

main_model = train(main_model, little_trainer_dataloader, device=device)

# ---------------------------------------------------------
# Data Sharing Strategy (Augmentation)
# ---------------------------------------------------------
alpha_percent = 10
total_shared_samples = len(little_trainer_dataset)
num_to_sample = int((alpha_percent / 100) * total_shared_samples)
augmented_train_loader = {}

for digit, private_dataset in train_digit_dataset.items():
    indices = random.sample(range(total_shared_samples), num_to_sample)
    shared_subset = Subset(little_trainer_dataset, indices)
    combined_dataset = ConcatDataset([private_dataset, shared_subset])
    augmented_train_loader[digit] = DataLoader(
        dataset=combined_dataset, shuffle=True, batch_size=32
    )

# ---------------------------------------------------------
# FedAMP Training Phase
# ---------------------------------------------------------
print("\n--- FedAMP Nodal Model Training ---")
num_communication_rounds = 3  # Increase this to 5 or 10 for full convergence
sigma_value = 1.0 # Hyperparameter to control attention stringency 

# Initialize 10 Personalized Cloud Models (U_i), starting from the warm-up model
U_models = [net().to(device) for _ in range(10)]
for u in U_models:
    u.load_state_dict(main_model.state_dict())

for round in range(num_communication_rounds):
    print(f"\n>> Communication Round {round + 1}/{num_communication_rounds} <<")
    all_models_weights = []
for digit in range(10):
    local_model = net()
    local_model.load_state_dict(main_model.state_dict())
    local_model = train_with_regularisation(model=local_model, global_model=main_model, dataloader=augmented_train_loader[digit], device=device)
    all_models_weights.append(local_model.state_dict())

    for digit in range(10):
        print(f"Training Node {digit}...")
        local_model = net().to(device)
        
        # 1. Download the Personalized Cloud Model (U_i) for this specific client
        local_model.load_state_dict(U_models[digit].state_dict())
        
        # 2. Train locally with Proximal Regularization pulling towards U_i
        # IMPORTANT: Using augmented_train_loader instead of train_digit_loader
        local_model = train_with_regularisation(
            model=local_model, 
            global_model=U_models[digit], 
            dataloader=augmented_train_loader[digit], 
            device=device
        )
        
        # 3. Save weights for the Server
        all_models_weights.append(local_model.state_dict())

    # 4. Server executes Attentive Message Passing
    print("Server executing FedAMP aggregation...")
    personalized_cloud_dicts = fedamp_averaging(all_models_weights, sigma=sigma_value)

    # 5. Update the U_i models for the next round
    for i in range(10):
        U_models[i].load_state_dict(personalized_cloud_dicts[i])

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
print("\n--- Final Evaluation ---")
# Because FedAMP creates personalized models, we test each client's specific model
total_accuracy = 0.0
for i in range(10):
    print(f"Testing Personalized Model for Node {i}:")
    # For a quick global check, we test against the entire test set. 
    # In a true local evaluation, you'd test against a local test set.
    acc = test(U_models[i], test_loader, device=device)
    total_accuracy += acc

print(f"\nAverage Global Accuracy across all Personalized Models: {total_accuracy / 10:.2f}%")