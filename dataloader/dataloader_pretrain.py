import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder

from custom_transforms import GaussianBlur, Solarization, TwoCropsTransform

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)

class CifarTransform(BaseTransform):
    def __init__(
        self,
        brightness: float=0.4,
        contrast: float=0.4,
        saturation: float=0.4,
        hue: float=0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        #TODO avoiding to chage the crop_size every time
        crop_size: int = 16,
    ):
        """Class that applies Cifar10/Cifar100 transformations.
        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 32.
        """

        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

class STLTransform(BaseTransform):
    def __init__(
        self,
        brightness: float=0.4,
        contrast: float=0.4,
        saturation: float=0.4,
        hue: float=0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        crop_size: int = 96,
    ):
        """Class that applies STL10 transformations.
        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 96.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )

class ImagenetTransform(BaseTransform):
    """
    The parameters are set according to MoCoV2
    """
    def __init__(
        self,
        brightness: float=0.4,
        contrast: float=0.4,
        saturation: float=0.4,
        hue: float=0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        crop_size: int = 224
    ) -> None:
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

def prepare_transform(dataset: str, **kwargs) -> Any:
    """Prepares transforms for a specific dataset. Optionally uses multi crop.
    Args:
        dataset (str): name of the dataset.
    Returns:
        Any: a transformation for a specific dataset.
    """

    if dataset in ["cifar10", "cifar100"]:
        return CifarTransform(**kwargs)
    elif dataset == "stl10":
        return STLTransform(**kwargs)
    elif dataset in ["imagenet1k", "imagenet100"]:
        return ImagenetTransform(**kwargs)

def prepare_datasets(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
) -> Dataset:
    """Prepares the desired dataset.
    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): training data directory
            to be appended to data_dir. Defaults to None.
        no_labels (Optional[bool], optional): if the custom dataset has no labels.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if data_dir is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_folder / "datasets"

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)
    
    if val_dir is None:
        val_dir = Path(f"{dataset}/val")
    else:
        val_dir = Path(val_dir)

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=True,
            transform=TwoCropsTransform(transform),
        )

        val_dataset = DatasetClass(
            data_dir / val_dir,
            train=False,
            download=download,
            transform=TwoCropsTransform(transform),
        )


    elif dataset == "stl10":
        train_dataset = STL10(
            data_dir / train_dir,
            split="unlabeled",
            download=True,
            transform=TwoCropsTransform(transform),
        )
        val_dataset = STL10(
            data_dir / val_dir,
            split="test",
            download=download,
            transform=TwoCropsTransform(transform),
        )
    elif dataset == "imagenet1k":
        train_dataset = ImageFolder(
            data_dir/ dataset/'train',
            transform=TwoCropsTransform(transform)
        )
        val_dataset = ImageFolder(
            data_dir/ dataset/'val',
            transform=TwoCropsTransform(transform)
        )


    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """
    random_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=random_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


if __name__ == '__main__':
  
    from utils import imshow
    train_transform = prepare_transform('cifar10')
    train_dataset = prepare_datasets('cifar10', train_transform)
    train_dataloader = prepare_dataloader(train_dataset, batch_size=1, num_workers=1)

    # get some random training images
    dataiter = iter(train_dataloader)
    (imagesq, imagesk), target= dataiter.next()


    # show images
    imshow(torchvision.utils.make_grid(imagesq))
    imshow(torchvision.utils.make_grid(imagesk))