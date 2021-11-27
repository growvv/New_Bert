import torch

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 2