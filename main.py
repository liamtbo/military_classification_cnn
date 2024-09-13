from dataset_class import CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torch.optim as optim
import sys
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
transformation = transforms.Compose([
    transforms.Resize((256,256)), # rets tensor uint8
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    # transforms uint8 to float for normalize
    transforms.ConvertImageDtype(torch.float),
    # expects float
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_data = CustomImageDataset(
    "./data/train_csv.csv", "./data/train_data", transform=transformation
)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

classes = ["tank", "aircraft"]

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(50_176, 120) # TODO hard coded checkout
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x)))) # (256-5+2*0)/1 + 1 = 252, 252/2 = 126
        x = self.pool(self.bn2(F.relu(self.conv2(x)))) # (126-5+2*0)/1 + 1 = 122, 122/2 = 61
        x = self.pool(self.bn3(F.relu(self.conv3(x)))) # (61-5+2*0)/1 + 1 = 57, 57/2 = 28
        # output is 64 x 28 x 28 = 50_176
        x = torch.flatten(x, 1) # 1 is so we flatten over dim=1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

print("Starting Training...")
for epoch in range(3):
    loss_sum = 0.0
    for i, data in enumerate(train_dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss
        if i % 10 == 0:
            print(f"step {i} loss: {loss_sum / 10}")
            loss_sum = 0

print("Training Complete")
print("Starting Testing...")
    
test_data = CustomImageDataset(
    "./data/test_csv.csv", "./data/test_data", transform=transformation
)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy() 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

test_loss = 0.0
for i, data in enumerate(test_dataloader):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    loss = criterion(outputs, labels)
    # imshow(torchvision.utils.make_grid(images))
    test_loss += loss
print(f"test loss: {test_loss / i}")


# train_images, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_images.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(train_labels)


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# imshow(torchvision.utils.make_grid(train_images))