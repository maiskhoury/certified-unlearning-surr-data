#!/usr/bin/env python
"""test for subsampled_hessian_unlearning function with ResNet18"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights

from src.forget import subsampled_hessian_unlearning
from src.loss import L2RegularizedCrossEntropyLoss

print("="*60)
print("Testing subsampled hessian unlearning (ResNet18)")
print("="*60)

# Small test configuration
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load tiny subset of CIFAR-10
transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = CIFAR10('./data', train=True, transform=transforms, download=True)

# Use tiny subsets for fast testing
n_total = 500    # Total dataset size
n_dss = 300      # D_ss subset (60%)
n_forget = 50    # Forget set (10%)

# Create subsets
indices = torch.randperm(len(train_dataset))[:n_total].tolist()
train_subset = Subset(train_dataset, indices)

dss_indices = indices[:n_dss]
forget_indices = indices[n_total-n_forget:n_total]  # Last 50 samples for forgetting

dss_dataset = Subset(train_dataset, dss_indices)
forget_dataset = Subset(train_dataset, forget_indices)

# Create dataloaders
dss_loader = DataLoader(dss_dataset, batch_size=32, shuffle=False)
forget_loader = DataLoader(forget_dataset, batch_size=32, shuffle=False)

print(f"\nDataset sizes:")
print(f"  Total (n1): {n_total}")
print(f"  D_ss (n2): {n_dss}")
print(f"  Forget (m): {n_forget}")

# ---- ResNet18 model modes (same as real_main.py return_model) ----
# 'linear': Flatten -> Linear(512, 10)  (uses pretrained feature dim, ignores conv layers)
# 'conv1':  layer4[1] -> avgpool -> Flatten -> Linear(512, 10)  (freeze early layers)
# 'conv2':  layer4[1] -> avgpool -> Flatten -> Linear(512, 10)  (all trainable)

resnet_mode = 'conv1'  # Change to 'conv1' or 'conv2' to test other modes
num_class = 10

print(f"\nUsing ResNet18 mode='{resnet_mode}'")
pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)

if resnet_mode == 'linear':
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(pretrained.fc.in_features, num_class)  # 512 -> 10
    )
    # CIFAR-10 images are 3x32x32 = 3072, but this mode expects flattened input
    # so input dim = 3072, but Linear expects 512... need to resize or use different input
    # Actually in real_main.py this mode just does Flatten + Linear(512, 10)
    # For CIFAR-10 (3072 input), this won't match. Let's adjust:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, num_class)  # Match CIFAR-10 flattened dim
    )
    print("  Note: Using Linear(3072, 10) for CIFAR-10 compatibility")
elif resnet_mode == 'conv1':
    model = nn.Sequential(
        pretrained.layer4[1],
        pretrained.avgpool,
        nn.Flatten(),
        nn.Linear(pretrained.fc.in_features, num_class)
    )
    # Freeze early layers (same as real_main.py)
    for idx, param in enumerate(model.parameters()):
        param.requires_grad = False
        if idx == 2:
            break
elif resnet_mode == 'conv2':
    model = nn.Sequential(
        pretrained.layer4[1],
        pretrained.avgpool,
        nn.Flatten(),
        nn.Linear(pretrained.fc.in_features, num_class)
    )
else:
    # Full ResNet18 with replaced fc layer for CIFAR-10
    pretrained.fc = nn.Linear(pretrained.fc.in_features, num_class)
    model = pretrained

model = model.to(device)

# Count parameters
d = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Trainable parameters (d): {d:,}")
print(f"  Total parameters: {total_params:,}")

# Train briefly (just to get non-random weights)
print("\nTraining model briefly...")
criterion = L2RegularizedCrossEntropyLoss(l2_lambda=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(2):  # Just 2 epochs for testing
    for data, target in dss_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  Epoch {epoch+1}/2 done")

# ---- Test unlearning with conjugate gradient (linear=False) ----
# ResNet is non-linear, so we must use conjugate gradient (no analytical Hessian)
print("\n" + "="*60)
print("Testing ResNet18 with conjugate gradient (linear=False)")
print("="*60)

try:
    umodel = subsampled_hessian_unlearning(
        model=model,
        dss_loader=dss_loader,
        forget_loader=forget_loader,
        criterion=criterion,
        device=device,
        n1=n_total,
        n2=n_dss,
        m=n_forget,
        eps=5.0,
        delta=1.0,
        eta=0.01,
        alpha=1.01,
        beta=1.0,
        gamma=1.0,
        L=1.1,
        d=d,
        C_constant=2.0,
        linear=False  # Conjugate gradient (required for ResNet)
    )
    print("\n SUCCESS: ResNet18 unlearning completed!")
except Exception as e:
    print(f"\n FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
