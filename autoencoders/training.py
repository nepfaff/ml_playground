from typing import Callable

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


def train_autoencoder(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: str,
    dataloader: DataLoader,
    epochs: int,
    print_loss: bool = True,
) -> None:
    """
    :param model: The model to train.
    :param optimizer: The optimizer to use.
    :param criterion: The loss function.
    :param dataloader: The data loader that contains the training data.
    :param epochs: The number of epochs to use.
    :param device: The device to train on. Either "cuda" or "cpu". The model should already be on this device.
    :param print_loss: If True, print the training loss for each epoch.
    """
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_curve = []

        for X, _ in dataloader:
            optimizer.zero_grad()

            X = X.to(device)

            # Forward pass
            X_reconstructed = model(X)
            loss = criterion(X_reconstructed, X)
            loss_curve += [loss.item()]

            # Backward pass
            loss.backward()
            optimizer.step()

        if print_loss:
            print(
                f"--- Iteration {epoch+1}: training loss = {torch.Tensor(loss_curve).mean().item():.4f} ---"
            )
