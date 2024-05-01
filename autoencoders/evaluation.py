from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader


def get_autoencoder_original_reconstructed_pairs(
    model: nn.Module, dataloader: DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param model: The model to train.
    :param dataloader: The data loader that contains the test data.
    :param device: The device to train on. Either "cuda" or "cpu". The model should already be on this device.
    :return: A tuple of (originals, reconstructed, labels) where both originals and reconstructed have shape (n,x,y)
        where (x,y) is the image dimension. labels is of shape (n,). The indices in all arrays must correspond.
    """
    original = []
    reconstructed = []
    labels = []
    with torch.no_grad():
        model.eval()

        for x, y in dataloader:
            x = x.to(device)

            x_reconstructed = model(x)

            original.append(x.cpu().detach().numpy())
            reconstructed.append(x_reconstructed.cpu().detach().numpy())
            labels.append(y.cpu().numpy())

    return np.array(original), np.array(reconstructed), np.array(labels)


def plot_original_reconstructed_per_class_grayscale(
    X_original: torch.Tensor,
    X_reconstruction: torch.Tensor,
    y: torch.Tensor,
    subplot_shape: Tuple[float, float],
    figsize: Tuple[float, float],
) -> None:
    """
    Plots one original instance next to its reconstruction per class as grayscale images.

    :param X_original: An array of original instances of shape (n,x,y) where (x,y) are the image dimensions.
    :param X_reconstruction: An array of reconstructed instances of shape (n,x,y) where (x,y) are the image dimensions.
        The array indices must correspond to the ones in `X_original`.
    :param y: An array of class labels of shape (n,).
    :param subplot_shape: Determines the subplot arrangement. The first element is the number of rows and the second the
        number of columns. This must account for enough subplots for both original and reconstructed images.
    :param figsize: The total figure size containing all subplots. A tuple of (x, y)
    """
    labels = torch.unique(y)
    fig = plt.figure(figsize=[figsize[0], figsize[1]])
    subplot_idx = 1
    for label in labels:
        first_label_idx = torch.where(y == label)[0][0]

        ax = fig.add_subplot(subplot_shape[0], subplot_shape[1], subplot_idx)
        ax.imshow(X_original[first_label_idx].reshape((28, 28)), cmap="gray")
        ax.set_title(f"Original of class {int(label)}")

        ax = fig.add_subplot(subplot_shape[0], subplot_shape[1], subplot_idx + 1)
        ax.imshow(X_reconstruction[first_label_idx].reshape((28, 28)), cmap="gray")
        ax.set_title(f"Reconstructed of class {int(label)}")

        subplot_idx += 2
