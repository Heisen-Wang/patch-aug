# dataloader

Dataloader for multiple datasets. The code is based on Python 3.8.5 and torch 1.8.1, and tested on workstation and **hdmfl** cluster.
When first run the dataloader, the **datasets** directory will be created. Directories of loading dataset will be made under the *datasets/*. The file tree is shown like this
```
Dataloader
|
Datasets
│      
│
└───imagenet1k
│   │  
│   └───train
│   |   │               
│   |   └───subfolder           
│   │       ... | file111.jpeg
│   └───val     | file112.jpeg

```

### Datasets

| Datasets  | Tasks  |Image size (train/test) |Download|
| :-------- | :------------: |:---------: |:--------------:|
| Imagenet1k  | Pretraining  | 1000k     | [ImageNet](https://image-net.org/challenges/LSVRC/2012/) |
| STL-10/Unlabeled | Pretraining         | 100k     | [STL-10](https://cs.stanford.edu/~acoates/stl10/) |
| Cifar10       | Pretraining/Classification         | 50k/10k    | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Cifar100       | Pretraining/Classification         | 50k/10k     | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| STL-10/Labeled       | Classification        | 5k/8k    | [STL-10](https://cs.stanford.edu/~acoates/stl10/) |
| Pascal VOC 2007     | Semantic segmentation         | 209/213     | [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) |
| CoCo      | Instance segmentation         | 118K/5K     | [CoCo](https://cocodataset.org/#download) |
| CoCo       | Object detection        | 118K/5K     | [CoCo](https://cocodataset.org/#download) |
| Cityscapes      | Semantic segmentation        | 5k     | [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) |


### Usage

0. Download data and move to dataset directory.

    Imagenet1k, CoCo, Cityscapes and Pascal VOC should be downloaded and move tho *datasets/* before runing the code. Those data has already stored in my scratch directory 
```
/p/scratch/haf/wang38/datasets

```

1. Run some dataloader test with the following command.


```
bash create_env.sh
```

```
bash run.sh
```
### Note
0. The transformations/augmentations for different dataset are according to MoCo V2, but they can be replaced with any transformations.
1. The transformations for Coco_detection are not complete.  Augmenting simultaneously images and  annotations are not achieved.
 