# Training requirements and error analysis

## Measured

| Quantity | Value |
|---|---|
| Training machine | Udacity workspace, Tesla T4, PyTorch 1.5.0 |
| Training data | 182 volumes, 2D slices, batch 8 |
| Time | 11 min 16 s for 10 epochs (64–67 s per epoch) |
| Same training on a Mac CPU (Intel i5, 16 GB) | about 6.5 h |
| Parameters / weight file | 31,030,723 / 124 MB |
| Inference | median 3.1 s per volume, Mac CPU |
| Disk | The script requires 2 GB free: checkpoint plus resume file are written each epoch |

Peak system memory and GPU memory were not measured.

## Possible efficiencies (not tested here)

* Mixed precision on the GPU may shorten training further.
* Stop early: the best validation loss came at epoch 4, so patience-based stopping could have saved several epochs.
* Use fewer base filters or levels: 31 million parameters is large for 64×64 inputs.
* Add a Dice term to cross-entropy to weight the small foreground more.
* Compare against a small 3D U-Net, which uses context across slices.
* Cache preprocessed slices if the dataset grows.

## Augmentation

Worth testing: small rotations, shifts and scaling (nearest-neighbour for labels), intensity gamma and scaling, mild noise and bias field.

Unlikely to help or harmful:

* Left–right, up–down or front–back flips: inputs arrive in one fixed orientation, and a front–back flip swaps the meaning of anterior and posterior.
* 90° or large rotations, which never occur in correctly oriented scans.
* Strong elastic warping or scaling: volume is the output, and the anatomy becomes implausible.
* Colour changes (images are greyscale) and cut-and-paste or mix-up of anatomy.

## Best and worst test volumes

| Volume | Dice | Manual / predicted total, mm³ |
|---|---|---|
| 334 (worst) | 0.823 | 2678 / 2571 |
| 298 | 0.845 | 2857 / 2748 |
| 070 | 0.854 | 3450 / 3125 |
| 015 | 0.857 | 2819 / 2664 |
| 165 | 0.930 | 3023 / 3027 |
| 008 | 0.931 | 3248 / 3206 |
| 212 | 0.933 | 3582 / 3674 |
| 180 (best) | 0.934 | 2678 / 2684 |

Observed: Dice is only weakly related to size (correlation 0.12). Volumes 334 and 180 have the same manual volume but are the worst and best cases. In case 334 the predicted posterior part is narrower than the manual one in the plotted slice. Case 252 has Dice 0.894 yet reads 450 mm³ low.

Hypotheses, not tested: thin tail sections lose a large share of their voxels to one-voxel boundary shifts; the anterior/posterior cut is a convention that is hard to see in the image; manual labels may vary at the borders. Confirming these needs repeat tracings and slice-level review by a radiologist.
