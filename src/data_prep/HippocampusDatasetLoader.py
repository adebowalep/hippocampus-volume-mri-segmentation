"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import nibabel as nib

from utils.utils import med_reshape, normalize_volume

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    # sorted so that a seeded train/val/test split is reproducible on any file system
    images = sorted(f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != "."))

    out = []
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # The starter used MedPy's load method; NiBabel returns the same voxel array for these
        # RAS NIfTI files and avoids the extra SimpleITK dependency. The spacing is kept so
        # that volumes can be reported in mm^3.
        image_nii = nib.load(os.path.join(image_dir, f))
        label_nii = nib.load(os.path.join(label_dir, f))
        image = image_nii.get_fdata(dtype=np.float32)
        label = np.asanyarray(label_nii.dataobj)

        if image.shape != label.shape:
            raise ValueError(f"{f}: image shape {image.shape} != label shape {label.shape}")
        if not np.isin(label, (0, 1, 2)).all():
            raise ValueError(f"{f}: unexpected label values {np.unique(label)}")
        if image.shape[1] > y_shape or image.shape[2] > z_shape:
            raise ValueError(f"{f}: shape {image.shape} does not fit the {y_shape}x{z_shape} patch")
        original_shape = image.shape

        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        image = normalize_volume(image)

        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here

        # TASK: med_reshape function is not complete. Go and fix it!
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape)).astype(np.float32)
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: the mask holds categorical class ids (0 background, 1 anterior, 2 posterior).
        # torch.nn.CrossEntropyLoss expects integer (long) class indices as its target;
        # med_reshape returns a float array, so we cast back to integers.

        out.append({"image": image, "seg": label, "filename": f,
                    "original_shape": original_shape,
                    "voxel_volume_mm3": float(abs(np.linalg.det(label_nii.affine[:3, :3])))})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
