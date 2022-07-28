import random
import numpy as np
import torch
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from data_utils import sections_generator, stitch

def pad_if_smaller(image, size, fill=0):
    min_size = min(image.size)
    if min_size < size:
        ow, oh = image.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        image = F.pad(image, (0, 0, padw, padh), fill=fill)
    return image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.float32)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomGaussianBlur:
    def __call__(self, image, target):
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return image, target

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        """Gaussian blur as a callable object.
        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.
        Args:
            x (torch.Tensor): an image in the tensor format.
        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.
        Args:
            img (Image): an image in the PIL.Image format.
        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        
        section_dimensions = (16,16)
        sections = sections_generator(x, section_dimensions)
        image = np.array(x)
        convert_tensor = T.ToTensor()
        transformed_sections = []
        for section in sections:
            section = T.ToPILImage()(section)
            transformed_section = self.base_transform(section)
            transformed_section = transformed_section.permute(1,2,0)
            transformed_sections.append(transformed_section)
        q = stitch(image, transformed_sections, section_dimensions)
        q = convert_tensor(q)

        transformed_sections = []
        for section in sections:
            section = T.ToPILImage()(section)
            transformed_section = self.base_transform(section)
            #transformed_section = np.array(section)
            transformed_section = transformed_section.permute(1,2,0)
            #transformed_section = np.transpose(transformed_section, 0, 2)
            transformed_sections.append(transformed_section)
        k = stitch(image, transformed_sections, section_dimensions)
        k = convert_tensor(k)
        return [q, k, convert_tensor(x)]   
