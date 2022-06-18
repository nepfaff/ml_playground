from typing import Tuple

import torch
import matplotlib.pyplot as plt


def plot_samples_per_class_histogram(y: torch.Tensor) -> None:
    """
    Plots a histogram showing the number of instances for each class.

    :param y: An array of class labels of shape (n,).
    """
    labels, counts = torch.unique(y, return_counts=True)
    plt.bar([str(label.item()) for label in labels], counts)
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.title("Number of samples per class")


def plot_one_sample_per_class_grayscale(
    X: torch.Tensor, y: torch.Tensor, subplot_shape: Tuple[float, float], figsize: Tuple[float, float]
) -> None:
    """
    Plots one instance per class as a grayscale image.

    :param X: An array of instances of shape (n, m) where m is the number of attributes.
    :param y: An array of class labels of shape (n,).
    :param subplot_shape: Determines the subplot arrangement. The first element is the number of rows and the second the
        number of columns.
    :param figsize: The total figure size containing all subplots. A tuple of (x, y)
    """
    labels = torch.unique(y)
    fig = plt.figure(figsize=[figsize[0], figsize[1]])
    for label in labels:
        first_label_idx = torch.where(y == label)[0][0]
        ax = fig.add_subplot(subplot_shape[0], subplot_shape[1], label + 1)
        ax.imshow(X[first_label_idx].reshape((28, 28)), cmap="gray")
        ax.set_title(f"Sample of class {label}")
