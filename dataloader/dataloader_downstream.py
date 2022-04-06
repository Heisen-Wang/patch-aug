import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, VOCSegmentation, Cityscapes

import custom_transforms
import custom_datasets

def collate_fn(batch):
    """
    Define collate function, this function is specifically used for coco detection.
    """
    return tuple(zip(*batch))

def prepare_transforms(dataset: str):
    """Prepares pre-defined train and test transformation pipelines for some datasets.
    Args:
        dataset (str): dataset name.
    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomCrop(size=32),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomCrop(size=96),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }
    voc_pipeline = {
        "T_train": custom_transforms.Compose(
            [
                custom_transforms.RandomHorizontalFlip(0.5),
                custom_transforms.RandomResize(256),
                custom_transforms.CenterCrop(224), 
                custom_transforms.RandomGaussianBlur(),
                custom_transforms.PILToTensor(),
                custom_transforms.ConvertImageDtype(torch.float32),
                custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": custom_transforms.Compose(
            [
                custom_transforms.RandomResize(256), 
                custom_transforms.CenterCrop(224), 
                custom_transforms.PILToTensor(),
                custom_transforms.ConvertImageDtype(torch.float32), 
                custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        ),
    }
    coco_detection_pipeline = {
        "T_train": transforms.Compose(
            [   
                transforms.Resize((256,256)),  
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [     
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        ),
    }
    coco_pipeline = {
        "T_train": custom_transforms.Compose(
             [    
                custom_transforms.RandomResize(256),
                custom_transforms.CenterCrop(224), 
                custom_transforms.PILToTensor(),
                custom_transforms.ConvertImageDtype(torch.float32),
                custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": custom_transforms.Compose(
            [
                custom_transforms.RandomResize(256),
                custom_transforms.CenterCrop(224), 
                custom_transforms.PILToTensor(),
                custom_transforms.ConvertImageDtype(torch.float32),
                custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        )
    }
    cityscapes_pipline = {
         "T_train": custom_transforms.Compose(
            [
                custom_transforms.RandomHorizontalFlip(0.5),
                custom_transforms.RandomResize(256),
                custom_transforms.CenterCrop(224), 
                custom_transforms.RandomGaussianBlur(),
                custom_transforms.PILToTensor(),
                custom_transforms.ConvertImageDtype(torch.float32),
                custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": custom_transforms.Compose(
            [
                custom_transforms.RandomResize(256),
                custom_transforms.CenterCrop(224), 
                custom_transforms.PILToTensor(),
                custom_transforms.ConvertImageDtype(torch.float32),
                custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
            ]
        )
    }

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "voc": voc_pipeline,
        "coco_detection": coco_detection_pipeline,
        "coco_segmentation": coco_pipeline,
        "cityscapes": cityscapes_pipline
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.
    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        data_dir Optional[Union[str, Path]]: path where to download/locate the dataset.
        train_dir Optional[Union[str, Path]]: subpath where the training data is located.
        val_dir Optional[Union[str, Path]]: subpath where the validation data is located.
    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_dir is None:
        sandbox_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_dir / "datasets"
    else:
        data_dir = Path(data_dir)

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if val_dir is None:
        val_dir = Path(f"{dataset}/val")
    else:
        val_dir = Path(val_dir)

    assert dataset in ["cifar10", "cifar100", "stl10", 
                        "voc","coco_segmentation","cityscapes","coco_detection"]

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            data_dir / val_dir,
            train=False,
            download=download,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            data_dir / train_dir,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            data_dir / val_dir,
            split="test",
            download=download,
            transform=T_val,
        )

    elif dataset == "voc":
        train_dir = data_dir / train_dir
        val_dir = data_dir / val_dir

        train_dataset = VOCSegmentation(
            train_dir, 
            image_set='train',
            download=True,
            transforms=T_train,
            )
        val_dataset = VOCSegmentation(
            val_dir, 
            image_set='trainval',
            download=True,
            transforms=T_val,
            )
    
    elif dataset == "coco_segmentation":
        train_dataset = custom_datasets.CoCoSegementation(
            data_dir / 'coco',
            split='train'
            )
        val_dataset = custom_datasets.CoCoSegementation(
            data_dir / 'coco',
            split='val'
        )
    elif dataset == "coco_detection":
        train_dataset = custom_datasets.CoCoDetection(
             data_dir / 'coco',
             transforms=T_train,
             split='train'
        )
        val_dataset = custom_datasets.CoCoDetection(
             data_dir / 'coco',
             split='val',
             transforms=T_val
        )


    elif dataset == "cityscapes":
        train_dataset = Cityscapes(
            data_dir / 'cityscapes', 
            split='train',
            mode= 'fine',
            target_type='semantic',
            transforms=T_train
        )
        val_dataset = Cityscapes(
            data_dir / 'cityscapes', 
            split='val',
            mode= 'fine',
            target_type='semantic',
            transforms=T_val
        )

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, detection:Optional[bool] = False
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.
    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn if detection else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn if detection else None
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.
    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train, T_val = prepare_transforms(dataset)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        download=download,
    )
    detection = True if dataset == "coco_detection" else False
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        detection=detection
    )
    return train_loader, val_loader


if __name__ == '__main__':
    from utils import decode_segmap, imshow
    
    train_dataloader, val_dataloader = prepare_data('stl10', batch_size=4, num_workers=4)
    # get some random training images
    dataiter = iter(train_dataloader)
    images, labels= dataiter.next()
    #labels = [decode_segmap(item,'pascal') for item in labels]
    imshow(torchvision.utils.make_grid(images))

