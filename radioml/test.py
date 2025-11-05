import torch

print("PyTorch verfügbar:", torch.__version__)
print("CUDA verfügbar:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Anzahl GPUs:", torch.cuda.device_count())
else:
    print("Keine GPU oder CUDA nicht verfügbar.")