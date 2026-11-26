"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape, normalize_volume, load_weights

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(load_weights(parameter_file_path, self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        if volume.ndim != 3:
            raise ValueError(f"Expecting a 3D volume, got shape {volume.shape}")
        if max(volume.shape[1:]) > self.patch_size:
            raise ValueError(f"In-plane size {volume.shape[1:]} exceeds the {self.patch_size} patch "
                             "the model was trained on; this is not a supported hippocampus crop")

        # same preprocessing as training: per-volume [0, 1] scaling, then zero padding
        x_size, y_size, z_size = volume.shape
        padded = med_reshape(normalize_volume(volume),
                             new_shape=(x_size, self.patch_size, self.patch_size))

        # crop the prediction back to the geometry of the input
        return self.single_volume_inference(padded)[:, :y_size, :z_size]

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        with torch.no_grad():
            for x in range(volume.shape[0]):
                data = torch.from_numpy(volume[x].astype(np.float32))[None, None].to(self.device)
                prediction = self.model(data)              # [1, num_classes, H, W]
                slices.append(torch.argmax(prediction, dim=1)[0].cpu().numpy().astype(np.uint8))

        return np.stack(slices, axis=0)
