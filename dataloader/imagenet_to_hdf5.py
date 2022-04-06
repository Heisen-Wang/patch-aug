import io
import os
import glob
import argparse

import h5py
import tqdm
import tarfile
import numpy as np
from pathlib import Path
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser(description='create hdf5 file from `tar')
    parser.add_argument('--target_dir', default="/home/qin/Work/data/hdf5", type=str)
    parser.add_argument('--source_dir', default="/home/qin/Work/project/datasets/imagenet_test/"
                        ,type=str)

def store_hdf5(images, labels, args):
    """Stores data to a hdf5 file
    Args:
        images (nparray): images array to store 
        labels (nparray): label array to store
    returns:
        a hdf5 file under specific dir
    """
    with h5py.File(Path(args.target_dir)/f"imagenet10k.h5", "w") as file:
        dataset = file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
        )
        labels = file.create_dataset(
            "labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
        )

def read_hdf5(filename, args):
    images, labels = [], []
    with h5py.File(Path(args.target_dir)/f"{filename}.h5", "r+") as file:
        images = np.array(file["/images"]).astype("uint8")
        labels = np.array(file["/labels"]).astype("uint8")
        return images, labels

def get_file_paths(args):
    return glob.glob(os.fspath(Path(args.target_dir)/'*.tar'))

def get_data_from_tar(file_path):
    images = []
    with tarfile.open(file_path) as file:
        for member in file.getmembers()[:-1]:
            f = file.extractfile(member)
            image = f.read()
            with Image.open(io.BytesIO(image)) as im:
                im = im.convert("RGB")
                im = im.resize((256,256), Image.BICUBIC)
                images.append(np.array(im))
        label = file_path.split('/')[-1].split('.')[0] 
        labels = [label]*len(file.getmembers()[:-1])
    return images, labels

def main():
    args = get_args_parser()
    file_paths = get_file_paths(args)
    for file_path in file_paths:
        images, labels = get_data_from_tar(file_path)

if __name__ == "__main__":
    
    main()
