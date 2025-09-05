import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
