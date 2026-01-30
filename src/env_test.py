import torch, torchvision, segmentation_models_pytorch as smp
import psutil, platform
import sys

print(f"操作系统: {sys.platform}")
print(f"Python版本: {platform.python_version()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
print(f"CPU核心数: {psutil.cpu_count(logical=False)}")
print(f"内存总量: {psutil.virtual_memory().total / 1e9:.2f}GB")
