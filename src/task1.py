# %%
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# %%
### --- GET DATA --- ###
NORMALIZE_MU = 0.5
NORMALIZE_STD = 0.5

# For MLP - flattened input
mlp_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((NORMALIZE_MU,), (NORMALIZE_STD,)),
    transforms.Lambda(lambda x: x.view(-1))
])


# For CNN - keep 2D structure
cnn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((NORMALIZE_MU,), (NORMALIZE_STD,))
])


# Load datasets for MLP
mlp_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mlp_transform)
mlp_test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=mlp_transform)

# Load datasets for CNN
cnn_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=cnn_transform)
cnn_test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=cnn_transform)


# %%
### --- CREATE DATALOADERS --- ###
BATCH_SIZE = 64

# MLP Dataloaders
mlp_train_loader = DataLoader(mlp_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
mlp_test_loader  = DataLoader(mlp_test_dataset , batch_size=BATCH_SIZE, shuffle=False)

# CNN Dataloaders
cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
cnn_test_loader  = DataLoader(cnn_test_dataset , batch_size=BATCH_SIZE, shuffle=False)


# %%
### --- EXAMPLE USAGE --- ###

# for batch_idx, (data, labels) in enumerate(mlp_train_loader):
#     # data shape for MLP: (batch_size, 784)
#     # labels shape: (batch_size,)
#     print(f"Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}")
#     print(data)
#     print(labels)
#     break  # Just show first batch



