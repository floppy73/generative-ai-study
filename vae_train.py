import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.vae import VAE
from common.dataset import BearDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

bear_imgs = torch.load("data/bear_imgs.pt")

batch_size = 32
img_size = 32  # image size: 32*32
latent_size = 8
epochs = 2000
learning_rate = 1e-3
train_size = int(len(bear_imgs) * 0.8)
val_size = len(bear_imgs) - train_size
    
bear_dataset = BearDataset(bear_imgs)
train_dataset, val_dataset = random_split(bear_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

vae = VAE(img_size, latent_size).to(device)
optimizer = optim.Adam(vae.parameters(), lr = learning_rate)
train_losses = []
val_losses = []

for i in tqdm(range(epochs)):
    train_loss_sum = 0.0
    train_cnt = 0
    val_loss_sum = 0.0
    val_cnt = 0

    vae.train()
    for x in train_dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        loss = vae.get_loss(x)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_cnt += 1
    
    loss_avr = train_loss_sum / train_cnt
    train_losses.append(loss_avr)

    vae.eval()
    with torch.no_grad():
        for x in train_dataloader:
            x = x.to(device)
            loss = vae.get_loss(x)
            val_loss_sum += loss.item()
            val_cnt += 1
    
    val_loss_avr = val_loss_sum / val_cnt
    val_losses.append(val_loss_avr)


x = np.arange(0, len(train_losses))
plt.plot(x, train_losses, label="training")
plt.plot(val_losses, label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
plt.clf()

with torch.no_grad():
    size = 36
    z = torch.randn(size, latent_size).to(device)
    images = vae.decoder(z).cpu()

grid_img = torchvision.utils.make_grid(
    images,
    nrow = 6,
    padding = 2,
    normalize = True
)

plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
