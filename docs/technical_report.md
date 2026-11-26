# Technical Report: Hippocampal Volume Quantification

Date: 2026-11-26

## 1. Executive summary

A 2D U-Net segments anterior and posterior hippocampus in cropped 1 mm brain MRI and reports volumes in mm³ as a DICOM image. The submitted model was trained in the Udacity workspace on a Tesla T4. On 39 held-out volumes it reached **mean Dice 0.897** (range 0.823–0.934), mean Jaccard 0.813, and mean absolute total-volume error 141 mm³ (4.2%). This is slightly below the course's "around 0.90" reference. Offline DICOM inference works on the three sample studies. In the Udacity workspace, Study1 was routed, analysed and the report stored back to Orthanc. The report was viewed in OHIF with its volumes. Nothing here is clinical validation.

## 2. Dataset

* 263 images and 262 labels downloaded; **260 pairs kept**.
* Excluded: 010 (a 512×512×241 CT-range image paired with a hippocampus-sized label), 118 (no label), 281 (a 512×512×31 CT-range image whose label shape differs).
* NIfTI-1, 1 mm isotropic, RAS. Retained images: 205 float32, 55 uint8. No duplicates found.

| Volume (mm³) | min | 5% | 25% | median | mean | 75% | 95% | max | SD |
|---|---|---|---|---|---|---|---|---|---|
| Total | 2382 | 2587 | 2950 | 3327 | 3295 | 3624 | 3979 | 4401 | 433 |
| Anterior | 1054 | | | 1725 | 1715 | | | 2593 | 307 |
| Posterior | 993 | | | 1579 | 1580 | | | 2169 | 229 |

Against the course nomogram (right side, women 52–71, read by eye as about 3000–4640 mm³), 26.5% fall below and none above. This is not a like-for-like comparison: the cohort, age, sex and tracing method differ or are unknown.

## 3. Method

* **Architecture:** the course starter's RecursiveUNet, unchanged: four down-sampling levels with skip connections, instance normalization, 3 output classes, 31.0 million parameters. It works on 2D sagittal slices padded to 64×64; slice predictions are stacked into a volume.
* **Preprocessing:** per-volume min–max to [0,1]; labels untouched; padding only.
* **Split:** 182 train / 39 validation / 39 test, by volume, seeded, spread across the volume range.
* **Training:** cross-entropy, Adam (2e-4), ReduceLROnPlateau, batch 8, 10 epochs, seed 42, no augmentation. Udacity Section 2 workspace: Tesla T4, PyTorch 1.5.0.

## 4. Training

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train loss | .0348 | .0131 | .0120 | .0113 | .0108 | .0105 | .0102 | .0103 | .0098 | .0092 |
| Val loss | .0151 | .0140 | .0133 | **.0132** | .0138 | .0138 | .0133 | .0137 | .0141 | .0141 |

* Total 11 minutes 16 seconds (64–67 s per epoch, including validation).
* The saved model is the lowest validation loss, epoch 4. Later epochs lowered training loss but not validation loss.
* The learning rate never changed from 2e-4.
* An earlier run on a Mac CPU (same code, split and seed, PyTorch 2.2.2) took about 6.5 hours and reached mean test Dice 0.895. Its files are not included in this repository. Its first attempt stopped in the first epoch, probably because the disk was full; resume support was added after that.

## 5. Evaluation (39 test volumes)

| Metric | Whole | Anterior | Posterior |
|---|---|---|---|
| Dice | 0.897 (SD 0.027, median 0.898) | 0.875 | 0.861 |
| Jaccard | 0.813 | 0.779 | 0.758 |
| Volume mean absolute error, mm³ | 141 | 127 | 120 |
| Volume mean absolute % error | 4.2 | 7.3 | 7.6 |
| Bias (predicted − manual), mm³ | −113 | −51 | −63 |
| 95% limits of agreement, mm³ | −398 to +171 | −348 to +246 | −340 to +214 |
| Pearson r | 0.94 | 0.85 | 0.82 |

Voxel sensitivity 0.882, specificity 0.995 (specificity is high mainly because background dominates). Reloading the saved model on the Mac reproduced every per-volume Dice from the GPU run exactly.

* The model **underestimates**: 28 of 39 total volumes are low, and the shortfall tends to grow with hippocampus size (correlation of error with true volume −0.31). Largest error: 450 mm³.
* Dice and volume can disagree: volume 252 has Dice 0.894 but reads 450 mm³ low. Both measures are needed.
* 3 of 39 predicted masks have more than one connected component (at most 9 voxels outside the largest). No post-processing is applied.
* Figures: `assets/test_best_worst.png`, `assets/volume_agreement.png`; per-volume values in `results/test_metrics_per_volume.csv`.
* The two runs differ by 0.002 in mean Dice, but two runs on one split do not give confidence intervals or a proper estimate of variation.

## 6. Clinical integration

| Step | Status |
|---|---|
| Select HippoCrop series, reject incompatible input | Done, tested offline |
| Segmentation and volumes on Studies 1–3 | Done: 3409, 2840, 3781 mm³ (no ground truth for these) |
| Secondary Capture report | Done: `examples/sample_report.dcm` (Study1) |
| Route Study1 and C-STORE the report to Orthanc (Udacity) | Done: `Success`; 3409 mm³ (`assets/pacs_inference_and_cstore.png`) |
| View the report in OHIF | Done: `assets/ohif_report.png` |

Median CPU inference on the Mac was 3.1 s per volume. The Udacity network is a simulation with no authentication, encryption or error recovery, and its sample studies pair crops with full-brain images from a different study.

## 7. Validation plan summary

Investigational volume estimate reviewed by a radiologist; multi-site, locked-model study of about 600 patients against a three-reader consensus; volume error, agreement, overlap, surface distance, repeatability and failure rate, by subgroup and scanner. The training data (one scanner, no dementia cohort) do not represent the proposed target population. Full text: `docs/validation_plan.pdf`.

## 8. Limitations

Single source and scanner; healthy and psychiatric adults only, ages unreported; no Alzheimer's or mild cognitive impairment data; right side only according to the course; possible subject overlap between splits cannot be ruled out; manual-label reliability unknown; no uncertainty estimate; no longitudinal repeatability, so no claim about progression; only Study1 was run through the PACS (Studies 2 and 3 offline only); 3D Slicer and segmentation overlay export not done.
