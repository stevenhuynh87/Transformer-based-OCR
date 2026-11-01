import torch
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
    free_memory = total_memory - allocated_memory

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Free Memory: {free_memory:.2f} GB")
else:
    print("No GPU found")
