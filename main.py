from dataset_class import CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

transformation = transforms.Compose([
    transforms.Resize((128,128)), # rets tensor uint8
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

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

classes = ["tank", "aircraft"]






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