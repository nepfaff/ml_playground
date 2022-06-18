from typing import Tuple, Optional

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms as tra


def get_MNIST_train_test_datasets(
    transforms: Optional[list] = None, filepath: str = "./"
) -> Tuple[DataLoader, DataLoader]:
    """
    :param transforms: A list of torchvision transforms to apply. `ToTensor()` is always applied.
    :param filepath: Path to download the datasets to.
    :return: A tuple of (train_dataset, test_dataset).
    """
    loader_transforms = [tra.ToTensor()]
    if transforms is not None:
        loader_transforms.extend(transforms)

    train_set = datasets.MNIST(
        filepath,
        train=True,
        download=True,
        transform=tra.Compose(loader_transforms),
    )
    test_set = datasets.MNIST(
        filepath,
        train=False,
        download=True,
        transform=tra.Compose(loader_transforms),
    )

    return train_set, test_set


def get_MNIST_train_validation_test_dataloaders(
    batch_size_train: int,
    batch_size_test: int,
    use_cuda: bool,
    train_split: float,
    transforms: Optional[list] = None,
    num_workers: int = 4,
    filepath: str = "./",
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    :param batch_size_train: The train and validation dataset batch size.
    :param batch_size_test: The test and validation dataset batch size.
    :param use_cuda: If True, copy the data directly into CUDA pinned memory.
    :param train_split: The fraction of the loaded train set that becomes the validation set in range [0,1]. Returns None
        for the validation_dataloader if this is 0.
    :param transforms: A list of torchvision transforms to apply. `ToTensor()` is always applied.
    :param num_workers: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main
        process.
    :param filepath: Path to download the datasets to.
    :return: A tuple of (train_dataloader, validation_dataloader, test_dataloader).
    """
    assert train_split >= 0.0 and train_split <= 1.0, "train_split must be in range [0,1]"

    train_validation_set, test_set = get_MNIST_train_test_datasets(transforms, filepath)

    if train_split != 0.0:
        # Split training set into training and validation sets
        instance_number = len(train_validation_set.targets)
        validation_instance_number = int(train_split * instance_number)
        train_instance_number = instance_number - validation_instance_number
        train_set, validation_set = random_split(
            train_validation_set, [train_instance_number, validation_instance_number]
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=batch_size_test,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )
    else:
        train_loader = DataLoader(
            train_validation_set,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )
        validation_loader = None
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    return train_loader, validation_loader, test_loader
