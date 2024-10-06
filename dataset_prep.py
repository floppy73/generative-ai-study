import torch
from torchvision import datasets, transforms

cifar100_train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.ToTensor())

cifar100_labels = cifar100_train_dataset.classes
bear_index = cifar100_labels.index("bear")

bear_imgs = []

for img, l in cifar100_train_dataset:
    if l == bear_index:
        bear_imgs.append(img)

bear_imgs = torch.stack(bear_imgs)
torch.save(bear_imgs, "data/bear_imgs.pt")
    
