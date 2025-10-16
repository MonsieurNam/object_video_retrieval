import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("cuda device count:", torch.cuda.device_count())
