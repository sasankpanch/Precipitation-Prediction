import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("M4 GPU Accelerated Performance Enabled")
else:
    device = torch.device("cpu")