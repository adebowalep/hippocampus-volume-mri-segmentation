"""
Various utility methods in this module
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from PIL import Image

# Tell Matplotlib to not try and use interactive backend
mpl.use("agg")

def mpl_image_grid(images):
    """
    Create an image grid from an array of images. Show up to 16 images in one figure

    Arguments:
        image {Torch tensor} -- NxWxH array of images

    Returns:
        Matplotlib figure
    """
    # Create a figure to contain the plot.
    n = min(images.shape[0], 16) # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2*rows, 2*cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using 
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: # plotting only 1st channel
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure

def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):
    """Logs data to Tensorboard

    Arguments:
        writer {SummaryWriter} -- PyTorch Tensorboard wrapper to use for logging
        loss {float} -- loss
        data {tensor} -- image data
        target {tensor} -- ground truth label
        prediction_softmax {tensor} -- softmax'd prediction
        prediction {tensor} -- raw prediction (to be used in argmax)
        counter {int} -- batch and epoch counter
    """
    writer.add_scalar("Loss",\
                    loss, counter)
    writer.add_figure("Image Data",\
        mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask",\
        mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map",\
        mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction",\
        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)

def save_numpy_as_image(arr, path):
    """
    This saves image (2D array) as a file using matplotlib

    Arguments:
        arr {array} -- 2D array of pixels
        path {string} -- path to file
    """
    plt.imshow(arr, cmap="gray") #Needs to be in row,col order
    plt.savefig(path)

def load_weights(path, device="cpu"):
    """torch.load restricted to tensors where the installed PyTorch supports it"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:       # PyTorch older than 1.13
        return torch.load(path, map_location=device)

def save_weights(state_dict, path):
    """Saves in the pre-1.6 file format where possible, so that the older PyTorch of the
    Udacity Section 3 workspace (1.4) can read a model trained with a newer release"""
    try:
        torch.save(state_dict, path, _use_new_zipfile_serialization=False)
    except TypeError:       # PyTorch 1.3 and older only write the old format
        torch.save(state_dict, path)

def normalize_volume(volume):
    """Per-volume min-max scaling to [0, 1]. MRI intensity units are arbitrary, so each volume
    is scaled by its own range. Used identically by training and inference.

    Arguments:
        volume {array} -- image volume (never a label volume)

    Returns:
        float32 array in [0, 1] (all zeros for a constant volume)
    """
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Image volume contains NaN or infinite values")
    low, high = float(volume.min()), float(volume.max())
    if high <= low:
        return np.zeros_like(volume)
    return (volume - low) / (high - low)

def med_reshape(image, new_shape):
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner

    Arguments:
        image {array} -- 3D array of pixel data
        new_shape {3-tuple} -- expected output shape

    Returns:
        3D array of desired shape, padded with zeroes
    """

    reshaped_image = np.zeros(new_shape)

    # TASK: write your original image into the reshaped image
    # Content stays at index (0, 0, 0); axes larger than new_shape are cropped. Values are
    # copied, never interpolated, so integer class labels survive unchanged and the
    # operation is reversed by slicing back to the original shape.
    overlap = tuple(slice(0, min(o, n)) for o, n in zip(image.shape, new_shape))
    reshaped_image[overlap] = image[overlap]

    return reshaped_image
