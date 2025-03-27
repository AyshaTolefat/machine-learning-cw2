import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from sklearn.cluster import KMeans
import numpy as np
import argparse
import os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Argument parser setup
parser = argparse.ArgumentParser(description="TPC-RP with SimCLR, SSL, and Active Learning")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_labels", type=int, default=4000)
parser.add_argument("--simclr_path", type=str, default="./SCAN/models/simclr_cifar10.pth")
parser.add_argument("--ssl_threshold", type=float, default=0.95)
args = parser.parse_args()

# Model definition
class ResNetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, 10)
        )

    def forward(self, x):
        return self.resnet(x)

# Adaptive Mixup function
def adaptive_mixup_data(model, x, y, alpha=0.4):
    """Apply Mixup only to uncertain samples."""
    with torch.no_grad():
        outputs = model(x.to(device))
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)  # Compute entropy
        threshold = entropy.median()  # Use median entropy as threshold
        mixup_mask = (entropy > threshold).to(device)  # Ensure boolean tensor is on CUDA

    if mixup_mask.any():  # Ensure we have valid samples for mixup
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=device)
        mixed_x = x.clone()
        mixed_x[mixup_mask] = lam * x[mixup_mask] + (1 - lam) * x[index[mixup_mask]]
        return mixed_x, y, y[index], lam, mixup_mask
    
    return x, y, y, 1, torch.zeros_like(y, dtype=torch.bool, device=device)

# Mixup loss function
def mixup_criterion(criterion, pred, y_a, y_b, lam, mixup_mask):
    mixed_indices = torch.where(mixup_mask)[0]
    normal_indices = torch.where(~mixup_mask)[0]
    
    mixed_loss = lam * criterion(pred[mixed_indices], y_a[mixed_indices]) + \
                 (1 - lam) * criterion(pred[mixed_indices], y_b[mixed_indices]) if mixed_indices.numel() > 0 else 0
    normal_loss = criterion(pred[normal_indices], y_a[normal_indices]) if normal_indices.numel() > 0 else 0
    
    return mixed_loss + normal_loss

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            mixed_inputs, targets_a, targets_b, lam, mixup_mask = adaptive_mixup_data(model, inputs, labels, alpha=0.4)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam, mixup_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets_a).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {100 * correct / total:.2f}% - Loss: {total_loss:.2f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        print(f"Val Acc: {100 * correct / total:.2f}%")
        scheduler.step()

# Evaluation function
def evaluate(model):
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor(), download=True),
        batch_size=256, shuffle=False
    )
    model.eval()
    correct, total = 0, 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")

# Run training and evaluation
full_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transforms.ToTensor(), download=True)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

model = ResNetWithDropout().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=args.epochs)
evaluate(model)















