from dataset_class import CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torch.optim as optim
import sys


transformation = transforms.Compose([
    transforms.Resize((32,32)), # rets tensor uint8 #TODO increase
    # transforms uint8 to float for normalize
    transforms.ConvertImageDtype(torch.float),
    # expects float
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_data = CustomImageDataset(
    "./data/train_csv.csv", "./data/train_data", transform=transformation
)
test_data = CustomImageDataset(
    "./data/test_csv.csv", "./data/test_data", transform=transformation
)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

classes = ["tank", "aircraft"]

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120) # TODO hard coded checkout
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 1 is so we flatten over dim=1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(2):
    loss_sum = 0.0
    for i, data in enumerate(train_dataloader):
        images, labels = data
        # print(images.size())
        # print(images.size())
        optimizer.zero_grad()
        outputs = net(images)
        # print(f"outputs: {outputs}\nlabels: {labels}")
        # sys.exit()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss
        if i % 10 == 0:
            print(f"step {i} loss: {loss_sum / 10}")

print("Training Complete")
    



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