import os
import numpy as np
import torch

from PIL import Image, ImageFile
from tqdm import trange
from typing import Callable, Optional
from pycocotools import mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import custom_transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CoCoSegementation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 split='train',
                 year='2017'):
        super().__init__()
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(root, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(root, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        # sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(_img, _target)
        elif self.split == 'val':
            return self.transform_val(_img, _target)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, image, target):
        composed_transforms = custom_transforms.Compose(
            [
            custom_transforms.RandomHorizontalFlip(0.5),
            custom_transforms.RandomResize(256),
            custom_transforms.CenterCrop(224), 
            custom_transforms.RandomGaussianBlur(),
            custom_transforms.PILToTensor(),
            custom_transforms.ConvertImageDtype(torch.float32),
            custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))
            ]    
            )

        return composed_transforms(image, target)

    def transform_val(self, image, target):

        composed_transforms = custom_transforms.Compose(
            [
            custom_transforms.RandomResize(256), 
            custom_transforms.CenterCrop(224), 
            custom_transforms.PILToTensor(),
            custom_transforms.ConvertImageDtype(torch.float32), 
            custom_transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))
            ]
            )

        return composed_transforms(image, target)


    def __len__(self):
        return len(self.ids)

class CoCoDetection(Dataset):
    def __init__(self, 
                root, 
                transforms,
                split='train',
                year='2017'):
        super().__init__()
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(root, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(root, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.transforms = transforms
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            self.ids = list(self.coco.imgs.keys())
            
        

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.img_dir, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)