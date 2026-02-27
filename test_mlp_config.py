"""
Quick test: Verify MLP model from config works with linear=False
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.loss import L2RegularizedCrossEntropyLoss
from src.forget import subsampled_hessian_unlearning
import logging
logging.basicConfig(level=logging.INFO)

# Simulate config settings
model_config = {
    'type': 'mlp',
    'hidden_sizes': [512, 256],  # From config
    'activation': 'relu',
    'bias': True  # From config
}

dim = 3072  # CIFAR-10 flattened
num_class = 10

# Create model using the same logic as real_main.py
def return_model(model_config, dim, num_class):
    if model_config['type'] == 'mlp':
        bias = model_config['bias']
        if model_config['hidden_sizes'] is not None:
            model_arr = [nn.Flatten()]
            curr_in = dim
            for size in model_config['hidden_sizes']:
                model_arr.append(nn.Linear(curr_in, size, bias=bias))
                if model_config['activation'] == 'relu':
                    model_arr.append(nn.ReLU())
                curr_in = size
            model_arr.append(nn.Linear(curr_in, num_class, bias=bias))
            model = nn.Sequential(*model_arr)
        else:
            model = nn.Sequential(nn.Flatten(), nn.Linear(dim, num_class, bias=bias))
        return model

model = return_model(model_config, dim, num_class)
print(f"\nModel architecture:\n{model}\n")

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")

# Create tiny dataset
n_samples = 100
X = torch.randn(n_samples, 3, 32, 32)
y = torch.randint(0, num_class, (n_samples,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
forget_loader = DataLoader(TensorDataset(X[:20], y[:20]), batch_size=32)
dss_loader = DataLoader(TensorDataset(X[:30], y[:30]), batch_size=32)

# Test with linear=False (conjugate gradient - works with non-linear models)
print("\nTesting with linear=False (conjugate gradient)...")
criterion = L2RegularizedCrossEntropyLoss(l2_lambda=0.01)

try:
    umodel = subsampled_hessian_unlearning(
        model=model,
        forget_loader=forget_loader,
        dss_loader=dss_loader,
        criterion=criterion,
        device='cpu',
        n1=100,
        n2=30,
        m=20,
        eps=1.0,
        delta=0.00001,
        eta=0.01,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        L=1.0,
        d=total_params,
        C_constant=2.0,
        linear=False  # Use conjugate gradient
    )
    print("SUCCESS: MLP with linear=False works!")
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "="*60)

