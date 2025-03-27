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

#Argument parser setup
parser = argparse.ArgumentParser(description="TPC-RP with SimCLR, SSL, and Active Learning")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_labels", type=int, default=4000)
parser.add_argument("--simclr_path", type=str, default="./SCAN/models/simclr_cifar10.pth")
parser.add_argument("--ssl_threshold", type=float, default=0.95)
args = parser.parse_args()

#SimCLR Feature Extractor
class SimCLRFeatureExtractor(nn.Module):
    def __init__(self, simclr_path):
        super().__init__()
        print(f"Loading SimCLR from {simclr_path}...")
        checkpoint = torch.load(simclr_path, map_location=device)
        new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if "contrastive_head" not in k}
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.load_state_dict(new_state_dict, strict=False)
        self.encoder.to(device).eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)

simclr_model = SimCLRFeatureExtractor(args.simclr_path)

#Data transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Model Definition
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

#Extract embeddings
@torch.no_grad()
def extract_embeddings(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    embeddings, labels = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        features = simclr_model(inputs).cpu().numpy()
        embeddings.append(features)
        labels.append(targets.numpy())
    return np.concatenate(embeddings).astype(np.float64), np.concatenate(labels)

#Entropy scoring
@torch.no_grad()
def compute_entropy_scores(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    scores = []
    model.eval()
    for inputs, _ in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        scores.extend(entropy.cpu().numpy())
    return np.array(scores)

#Simple SSL
@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_loader, threshold=0.95):
    pseudo_data = []
    model.eval()
    for inputs, _ in unlabeled_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        mask = max_probs >= threshold
        if mask.sum() > 0:
            pseudo_data.extend([(inp.cpu(), lbl.cpu()) for inp, lbl, keep in zip(inputs, pseudo_labels, mask) if keep])
    return pseudo_data

#Training
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {100*correct/total:.2f}% - Loss: {total_loss:.2f}")

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

#Evaluation
@torch.no_grad()
def evaluate(model):
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10(root="./data", train=False, transform=basic_transform, download=True),
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

#Active Learning with Enhanced Selection
full_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=basic_transform, download=True)
labels_array = np.array(full_dataset.targets)
initial_indices = [idx for c in range(10) for idx in np.where(labels_array == c)[0][:args.num_labels // 10]]
selected_indices = set(initial_indices)

NUM_AL_ITERS = 5
PER_ROUND = (args.num_labels * 2) // NUM_AL_ITERS

for al_iter in range(NUM_AL_ITERS):
    print(f"\nAL Iteration {al_iter + 1}/{NUM_AL_ITERS}")
    labeled_dataset = Subset(torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transform), list(selected_indices))
    unlabeled_indices = list(set(range(len(full_dataset))) - selected_indices)
    unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

    #Feature extraction & scoring
    embeddings, _ = extract_embeddings(unlabeled_dataset)
    kmeans = KMeans(n_clusters=10, random_state=al_iter).fit(embeddings)
    entropy_scores = compute_entropy_scores(ResNetWithDropout().to(device), unlabeled_dataset)

    #Combined typicality (distance to center) + entropy
    scores = []
    for i in range(len(embeddings)):
        cluster_id = kmeans.predict([embeddings[i].astype(np.float64)])[0]
        distance = np.linalg.norm(embeddings[i] - kmeans.cluster_centers_[cluster_id])
        combined_score = entropy_scores[i] + (1 / (distance + 1e-6))
        scores.append((unlabeled_indices[i], combined_score))
    top_samples = sorted(scores, key=lambda x: -x[1])[:PER_ROUND]
    selected_indices.update([idx for idx, _ in top_samples])

    #Data split
    final_labeled = Subset(torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transform), list(selected_indices))
    train_size = int(0.9 * len(final_labeled))
    val_size = len(final_labeled) - train_size
    train_dataset, val_dataset = random_split(final_labeled, [train_size, val_size])

    #SSL pseudo-labels
    pseudo_loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)
    ssl_model = ResNetWithDropout().to(device)
    pseudo_labels = generate_pseudo_labels(ssl_model, pseudo_loader, threshold=args.ssl_threshold)
    if pseudo_labels:
        pseudo_inputs, pseudo_targets = zip(*pseudo_labels)
        pseudo_dataset = torch.utils.data.TensorDataset(torch.stack(pseudo_inputs), torch.tensor(pseudo_targets))
        train_dataset = ConcatDataset([train_dataset, pseudo_dataset])

    #Training
    model = ResNetWithDropout().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=args.epochs)
    evaluate(model)
