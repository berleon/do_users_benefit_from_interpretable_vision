"""data loading."""

from __future__ import annotations

import abc
import collections
from collections import OrderedDict
import copy
import csv
from glob import glob
import itertools
import os
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import skimage.segmentation as seg
from skimage.transform import AffineTransform, rotate
from skimage.transform import resize
import skimage.util
import torch
from torch.utils.data import (
    DataLoader,
    Dataset as TorchDataset,
    RandomSampler,
    Subset as TorchSubset,
)
from torchvision import transforms
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
import tqdm.auto as tqdm_auto
import two4two.pytorch


class Dataset(TorchDataset, collections.abc.Sized, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def label_names(self) -> Sequence[str]:
        pass

    def get_label_index(self, label_name: str) -> int:
        return self.label_names.index(label_name)

    def get_label_mapping(self) -> dict[str, int]:
        return {name: self.get_label_index(name) for name in self.label_names}


class DeterministicShuffle(Dataset):
    def __init__(self, dataset: Dataset, seed: int = 0):
        self.dataset = dataset
        np.random.seed(seed)
        self.indices = np.random.choice(len(dataset), len(dataset), replace=False)

    @property
    def label_names(self) -> Sequence[str]:
        return self.dataset.label_names

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.dataset)


class CIFAR10(TorchCIFAR10, Dataset):
    @property
    def label_names(self) -> Sequence[str]:
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = super().__getitem__(idx)
        return img, torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)


class Subset(TorchSubset, Dataset):
    dataset: Dataset

    @property
    def label_names(self) -> Sequence[str]:
        return self.dataset.label_names


def torch_img_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Converts a torch image to a numpy array."""
    img_np = img.cpu().detach().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    return (255 * img_np.transpose(1, 2, 0)).astype(np.uint8)


class UniformNoise:
    """Adds uniform noise of ``(-0.5*scale, 0.5*scale)``."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Adds uniform noise to ``x``."""
        return x + self.scale * (torch.rand_like(x) - 0.5)


FilterFunc = Callable[[np.ndarray], np.ndarray]


DatasetItem = Tuple[Union[PIL.Image.Image, torch.Tensor], Sequence[np.ndarray]]


def _get_eye_pos(img: np.ndarray) -> np.ndarray:
    from mlxtend.image import extract_face_landmarks  # type: ignore

    landmarks = extract_face_landmarks(img)
    if landmarks is None:
        raise ValueError("No Face detected")
    left_eye = np.arange(36, 42)
    right_eye = np.arange(42, 48)
    return np.array(
        [
            landmarks[left_eye].mean(0),
            landmarks[right_eye].mean(0),
        ]
    )


class Two4Two(two4two.pytorch.Two4Two, Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Any = transforms.ToTensor(),
        return_attributes: Sequence[str] = ["obj_name"],
        return_segmentation_mask: bool = True,
        expand_segmentation_mask: int = 0,
    ):
        self._expand_segmentation_mask = expand_segmentation_mask
        super().__init__(
            root_dir, split, transform, return_attributes, return_segmentation_mask
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        if not self._return_segmentation_mask:
            return super().__getitem__(index)

        img, mask, label = super().__getitem__(index)
        return (
            img,
            seg.expand_labels(mask.numpy(), self._expand_segmentation_mask),
            label,
        )

    @property
    def label_names(self) -> Sequence[str]:
        return self.get_label_names()


DATALOADERS_AND_DATASETS = Tuple[
    DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset
]


def load_two4two_split(
    dataset_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    dataset_kwargs: Dict[str, Any] = {},
) -> Tuple[DataLoader, Two4Two]:
    """Factory to load the dataset with the appropiate default settings.

    Args:
        dataset (str): Dataset name
        dataset_dir (str): Root dir of the dataset
        split (str): train / test / valdation split
        batch_size (int): Batch size
        num_workers (int): Number of workers to paralize the data loader

    Returns:
        A tuple of ``(loader, dataset)``.
    """
    dataset_kwargs = copy.copy(dataset_kwargs)
    dataset_kwargs["return_segmentation_mask"] = dataset_kwargs.get(
        "return_segmentation_mask", False
    )

    dataset = Two4Two(
        dataset_dir,
        split=split,
        transform=Compose(
            [
                ToTensor(),
                UniformNoise(scale=1.0 / 256),
            ]
        ),
        **dataset_kwargs,
    )
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset


def load_datasets(
    dataset: str,
    dataset_dir: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 4,
    dataset_kwargs: Dict[str, Any] = {},
) -> DATALOADERS_AND_DATASETS:
    """Factory to load the dataset with the appropiate default settings.

    Args:
        dataset (str): Dataset name
        dataset_dir (str): Root dir of the dataset
        batch_size (int): Batch size
        image_size (int): Size of the image
        num_workers (int): Number of workers to paralize the data loader

    Returns:
        A tuple of ``(train_loader, val_loader, test_loader, train_set, test_set, val_set)``.

    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    train_set: Dataset
    test_set: Dataset
    val_set: Dataset

    if dataset == "mice":
        transform = MiceImgAugTransform(image_size)
        train_set = MiceDataset(dataset_dir, transform=transform, split="train")
        val_set = MiceDataset(dataset_dir, transform=transform, split="validation")
        test_set = MiceDataset(dataset_dir, transform=transform, split="test")

        train_loader = DataLoader(
            train_set, batch_size, shuffle=True, num_workers=num_workers
        )
        val_random_gen = torch.manual_seed(0)
        val_loader = DataLoader(
            val_set,
            batch_size,
            sampler=RandomSampler(val_set, generator=val_random_gen),
            num_workers=num_workers,
        )
        test_random_gen = torch.manual_seed(0)
        test_loader = DataLoader(
            test_set,
            batch_size,
            sampler=RandomSampler(test_set, generator=test_random_gen),
            num_workers=num_workers,
        )
    elif dataset == "celeba":
        train_set = CelebA(
            dataset_dir,
            partition="train",
            transform=Compose(
                [
                    RandomHorizontalFlip(0.5),
                    CenterCrop(148),
                    Resize(image_size),
                    ToTensor(),
                    UniformNoise(scale=1.0 / 256),
                ]
            ),
        )

        test_set = CelebA(
            dataset_dir,
            partition="test",
            transform=Compose(
                [
                    CenterCrop(148),
                    Resize(image_size),
                    ToTensor(),
                ]
            ),
        )

        val_set = CelebA(
            dataset_dir,
            partition="validation",
            transform=Compose(
                [
                    CenterCrop(148),
                    Resize(image_size),
                    ToTensor(),
                ]
            ),
        )

        train_loader = DataLoader(
            train_set, batch_size, shuffle=False, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_set, batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_set, batch_size, shuffle=False, num_workers=num_workers
        )
    elif dataset == "cifar-test":
        train_set = CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=Compose(
                [
                    RandomHorizontalFlip(0.5),
                    Resize(image_size),
                    ToTensor(),
                    UniformNoise(scale=1.0 / 256),
                ]
            ),
        )
        test_set = CIFAR10(
            dataset_dir,
            train=False,
            download=True,
            transform=Compose([Resize(image_size), ToTensor()]),
        )
        val_set = test_set

        train_set = Subset(train_set, list(range(100)))
        val_set = Subset(val_set, list(range(100)))
        test_set = Subset(test_set, list(range(100)))

        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=1)
    elif dataset.startswith("two4two"):
        train_loader, train_set = load_two4two_split(
            dataset_dir,
            "train",
            batch_size,
            num_workers,
            dataset_kwargs,
        )
        test_loader, test_set = load_two4two_split(
            dataset_dir,
            "test",
            batch_size,
            num_workers,
            dataset_kwargs,
        )

        val_loader, val_set = load_two4two_split(
            dataset_dir,
            "validation",
            batch_size,
            num_workers,
            dataset_kwargs,
        )
    else:
        raise Exception()
    return train_loader, val_loader, test_loader, train_set, test_set, val_set


def collect_data_and_labels(
    loader: DataLoader,
    n_batches: int,
    start: int = 0,
    dataset_kwargs: Dict[str, Any] = {},
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    dataset_kwargs["return_segmentation_mask"] = dataset_kwargs.get(
        "return_segmentation_mask", False
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        batches = []
        for idx, batch in tqdm_auto.tqdm(
            enumerate(itertools.islice(loader, start, start + n_batches))
        ):
            batches.append(batch)

        if dataset_kwargs["return_segmentation_mask"]:
            imgs = torch.cat([batch[0] for batch in batches])
            masks = torch.cat([batch[1] for batch in batches])
            labels = torch.cat([batch[2] for batch in batches])
            return imgs, masks, labels
        else:
            imgs = torch.cat([batch[0] for batch in batches])
            labels = torch.cat([batch[1] for batch in batches])
            return imgs, labels
