"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    a, b = a > 0, b > 0
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0      # both masks empty: perfect agreement by convention
    return float(2.0 * np.logical_and(a, b).sum() / total)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0      # both masks empty: perfect agreement by convention
    return float(np.logical_and(a, b).sum() / union) #

def Sensitivity(pred, truth):
    """Fraction of true foreground voxels that were found (low value = under-segmentation).
    NaN when the reference mask is empty."""
    pred, truth = pred > 0, truth > 0
    positives = truth.sum()
    return float(np.logical_and(pred, truth).sum() / positives) if positives else float("nan")

def Specificity(pred, truth):
    """Fraction of true background voxels left as background. Dominated by background in
    medical images, so it is close to 1 even for mediocre masks. NaN if no background."""
    pred, truth = pred > 0, truth > 0
    negatives = (~truth).sum()
    return float(np.logical_and(~pred, ~truth).sum() / negatives) if negatives else float("nan")
