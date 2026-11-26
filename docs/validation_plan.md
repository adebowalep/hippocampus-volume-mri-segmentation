# HippoVolume.AI: Clinical Validation Plan (draft)

## 1. Intended use

HippoVolume.AI is an **investigational** tool that estimates the anterior, posterior and total volume (mm³) of one hippocampus from a cropped 1 mm isotropic T1-weighted brain MRI (the "HippoCrop" series), and returns a picture report to the PACS. A radiologist reviews the mask and decides whether to use the number. It does not diagnose Alzheimer's disease, stage it, or establish progression, and it is not a basis for treatment. The model was built on right-hippocampus data according to the project description; the validation must confirm side handling and the tool must not be used on the left side until that side is validated.

**Target population (proposed claim):** adults aged 55–90 referred for MRI because of memory complaints or suspected neurodegenerative disease, in radiology and memory-clinic settings, with no large lesion, resection or gross artefact in the medial temporal lobe.

## 2. Training data and its ground truth

* Medical Segmentation Decathlon, Task04 Hippocampus; 260 curated crops (182 train, 39 validation, 39 test).
* Published description (Simpson et al., arXiv:1902.09063): T1 MPRAGE, 1 mm³, one Philips Achieva scanner at Vanderbilt University Medical Center; 90 healthy adults and 105 adults with a non-affective psychotic disorder; manual tracing following published protocols. Ages, sex, number of tracers and rater agreement are not reported there. (The course text calls the images T2; the published description says T1.)
* Consequently the training cohort **does not represent the proposed target population**: no Alzheimer's or mild-cognitive-impairment cohort, one scanner, one site. Files have no subject identifiers, so subject-level separation of the offline test split cannot be proven.

## 3. Current evidence (offline, not clinical)

<!--RESULTS-->
On 39 held-out research crops from the same source as the training data: mean Dice 0.897 (SD 0.027, range 0.823–0.934), mean Jaccard 0.813; anterior / posterior Dice 0.875 / 0.861; total-volume mean absolute error 141 mm³ (4.2%), bias -113 mm³, 95% limits of agreement -398 to +171 mm³. This is internal testing only. It says nothing about other sites, scanners or patient groups, and it is not clinical validation.
<!--/RESULTS-->

## 4. Reference standard for validation

1. Three neuroradiologists, blinded to the algorithm and to each other, segment every case with a written protocol (EADC-ADNI harmonised hippocampal protocol), including an explicit anterior/posterior boundary at the uncal apex.
2. Consensus mask by majority vote; cases with pairwise Dice below 0.80 or volume spread above 10% go to adjudication by a fourth senior reader.
3. 10% of cases repeated after ≥ 4 weeks for intra-rater variability.
4. Quality control of orientation, spacing, completeness and mask alignment before locking the reference.

Manual tracing is itself uncertain. Inter- and intra-rater variability sets the ceiling of what can be demonstrated, so algorithm–reference disagreement is judged against reader–reader disagreement, and results are reported against each reader as well as the consensus.

## 5. Study design

* Retrospective, multi-site, locked model (weights identified by SHA-256 on every report), data never used in development.
* At least 5 sites, 3 vendors, 1.5 T and 3 T; consecutive eligible patients plus enrichment so each key subgroup has ≥ 50 cases: cognitively normal, MCI, AD dementia, other dementias; age bands 55–64, 65–74, 75–90; both sexes; diverse ancestry. Planned total about 600 patients, with the final number set by a power calculation on the primary endpoints using pilot variance.
* Test the whole pipeline, including HippoCrop, because crop errors reach the model.
* **Repeatability:** 50 volunteers scanned twice the same day with repositioning, and a subset on two scanners.

## 6. Endpoints and proposed acceptance criteria (to be agreed with clinicians before unblinding)

| Measure | Why | Proposed criterion |
|---|---|---|
| Absolute and relative total-volume error | the clinical output is a volume | mean absolute percentage error ≤ 5%, 95% CI upper bound |
| Bias and Bland–Altman limits of agreement | systematic over/under-estimation | bias within ±2%; limits no wider than between readers |
| ICC(2,1) for volume against consensus | agreement, not just correlation | ≥ 0.90 (lower 95% bound) |
| Dice and Jaccard: total, anterior, posterior | spatial correctness; equal volumes can come from wrong masks | median total Dice ≥ 0.85; non-inferior to inter-reader Dice (margin 0.03) |
| Surface distance (95th percentile Hausdorff) | local boundary errors | non-inferior to readers |
| Test–retest coefficient of variation | needed before any longitudinal claim | ≤ 2%, smaller than the annual change of interest |
| Failure and rejection rate | silent failures are the main risk | gross failure (Dice < 0.70) ≤ 2%, all listed and reviewed |

Overlap alone is not sufficient: Dice can look good while volume is biased, and a volume can be right for the wrong voxels. All endpoints are reported with confidence intervals and for every subgroup, site and scanner.

**Normative comparison.** Any percentile statement needs norms matched on age, sex, side, head size, scanner and segmentation protocol. Published nomograms such as Nobis et al. (UK Biobank; HippoFit calculator) use a different method and cannot be applied to this output without bridging evidence. Accuracy against disease status (e.g. sensitivity/specificity for AD) would be a separate claim requiring its own study with clinical follow-up.

## 7. Data the algorithm may operate on

Accept only: MR, T1-weighted 3D acquisition, HippoCrop series, 1 mm isotropic (±5%), axial orientation as delivered by HippoCrop, in-plane size ≤ 64 × 64, evenly spaced slices, adult within the validated age range. The software currently checks modality, series name, spacing, orientation, size and slice regularity; it does **not** yet check sequence type, age, side or image quality, which must be added before validation. Everything else is out of scope until shown otherwise: other contrasts (T2, FLAIR), thick slices, paediatric brains, tumours, surgery, large strokes, severe motion or metal artefact, unseen vendors or protocols. Severely atrophic hippocampi are in the target population but under-represented in training, so errors there are a named risk.

## 8. Safety, limitations and monitoring

* The clinician remains responsible; the report shows the mask so it can be rejected. Automatic flags (empty region, mask at crop edge, volume outside the training range) catch only obvious failures. The model gives no calibrated uncertainty.
* Hazards to analyse (ISO 14971): wrong series or patient, wrong side, wrong spacing, plausible but wrong volume, over-reliance.
* The simulated Udacity network is not a clinical workflow: no authentication, encryption, audit trail or error recovery, and the sample studies combine crops with full-brain series from a different study.
* After release: log inputs and versions, collect reader corrections, monitor volume and rejection-rate drift per scanner, and revalidate after any change to model, crop tool or protocol.
* Independent prospective validation, ethics approval and regulatory clearance are required before any diagnostic use.
