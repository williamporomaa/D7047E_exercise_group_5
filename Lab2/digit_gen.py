import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

def generate_digit(G, digit, Z_dim, num_classes=10):
    G.eval()

    # Create a batch of noise
    z = torch.randn(1, Z_dim).to(device)

    # Create the label
    label = torch.tensor([digit], dtype=torch.long).to(device)
    label_onehot = F.one_hot(label, num_classes=num_classes).float()

    # Generate an image
    with torch.no_grad():
        generated_img = G(z, label_onehot)

    return generated_img

def generate_grid(G, Z_dim, device, num_classes=10, digits_to_print=10):
    G.eval()

    all_images = []
    all_labels = []

    for digit in range(num_classes):
        z = torch.randn(digits_to_print, Z_dim).to(device)
        labels = torch.full((digits_to_print,), digit, dtype=torch.long).to(device)
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float()

        with torch.no_grad():
            generated_imgs = G(z, labels_onehot)

        all_images.append(generated_imgs.cpu())
        all_labels.append(labels.cpu())

    all_images = torch.cat(all_images, dim=0)

    plot_grid(all_images)

def plot_grid(images, num_classes=10, digits_to_print=10, img_shape=(28, 28)):
    fig, axes = plt.subplots(num_classes, digits_to_print, figsize=(digits_to_print, num_classes))

    idx = 0
    for i in range(num_classes):
        for j in range(digits_to_print):
            ax = axes[i, j]
            img = images[idx].view(*img_shape)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            idx += 1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()