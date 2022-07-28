import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError
    label_mask = torch.squeeze(label_mask)
    r = label_mask.clone()
    g = label_mask.clone()
    b = label_mask.clone()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def imshow(img):
   # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#TODO build a generator instead of returning a list
def sections_generator(image, section_dimensions):
    """
    Return should be the list of patches for one single image
    """
    
    # convert_tensor = transforms.ToTensor()
    # image = convert_tensor(image)
    # image = torch.as_tensor(image, dtype=torch.uint8)
    # image = torch.transpose(image, 0, 2)
    
    image = np.array(image)
    sections = []
    image_dimensions = (image.shape[0], image.shape[1])
    slices = compute_ranges(image_dimensions, section_dimensions)

    for i in range(len(slices)):
        sections.append(image[tuple(slices[i])])

    return sections

def compute_ranges(tensor_dimensions, section_dimensions):
    
    size_of_ranges = len(section_dimensions)*len(section_dimensions)
    ranges = []
    size_ranges = compute_size_ranges(tensor_dimensions[0], section_dimensions[0])
    
    for i in range(len(section_dimensions)):
        for j in range(len(section_dimensions)):
            ranges.append([size_ranges[i],size_ranges[j]])
    return ranges
        
    

def compute_size_ranges(total_size, section_size):
        """
        Create a list of all slice objects; one object per section

        """

        return [
            compute_size_range(i, total_size, section_size)
            for i in range(0, total_size, section_size)
        ]
    
def compute_size_range(i, total_size, section_size):
        """
        Create a slice object given the parameters. The slice object
        represents a range of indices.

        """

        # Range end: Taking the minimum of the two values ensures
        # that the end never exceeds 'total_size'. If the range
        # does exceed 'total_size', it is slided back to make sure
        # it fits. This causes overlap with a previous range.
        end = min(total_size, i + section_size)

        # Range start. Automatically adjusted to slide the range
        # backwards if necessary.
        start = end - section_size

        return slice(start, end)

def stitch(image, sections, section_dims):
    # Image dimensions
    image_dimensions = (image.shape[0], image.shape[1], image.shape[2])

    # Initialize stitched image
    stitched_mask = np.ones(image_dimensions, dtype=np.float32)

    # Slices generator
    slices = compute_ranges((image.shape[0],image.shape[0]),section_dims)

    # For each section,
    for i in range(len(sections)):
        # Get the next slice
        current_slice = slices[i]

        # Add the current section to the stitched image
        stitched_mask[tuple(current_slice)] = sections[i]

    return stitched_mask