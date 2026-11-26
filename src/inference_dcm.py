"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import copy
import json
import hashlib
import datetime
import time
import shutil
import argparse
import subprocess

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

ALGORITHM_NAME = "HippoVolume.AI"
ALGORITHM_VERSION = "1.0"
SERIES_DESCRIPTION_FOR_INFERENCE = "HippoCrop"
PATCH_SIZE = 64
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SRC_DIR, "assets", "Roboto-Regular.ttf")
COLOURS = {1: (230, 50, 50), 2: (60, 200, 90)}       # anterior red, posterior green

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    if not dcmlist:
        raise ValueError("No DICOM instances to build a volume from")
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = copy.deepcopy(sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)[0])

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_voxel_volume_mm3(dcmlist):
    """Physical size of one voxel: in-plane PixelSpacing times the distance between adjacent
    slice positions (SliceThickness only if there is a single slice).

    Raises ValueError for gaps, duplicates or inconsistent spacing, because the volume
    measurement would then be wrong.
    """
    ordered = sorted(dcmlist, key=lambda d: d.InstanceNumber)
    row_mm, col_mm = (float(v) for v in ordered[0].PixelSpacing)
    if len(ordered) > 1:
        orientation = np.array(ordered[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        positions = np.array([np.dot(np.array(d.ImagePositionPatient, dtype=float), normal)
                              for d in ordered])
        steps = np.diff(positions)
        if np.any(steps == 0) or not np.allclose(steps, steps[0], atol=1e-3):
            raise ValueError("Slice positions are duplicated, missing or unevenly spaced")
        slice_mm = abs(float(steps[0]))
    else:
        slice_mm = float(ordered[0].SliceThickness)
    return row_mm * col_mm * slice_mm

def get_predicted_volumes(pred, voxel_volume_mm3=1.0):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

        voxel_volume_mm3 {float} -- physical volume of one voxel

    Returns:
        A dictionary with respective volumes in mm^3
    """

    # TASK: Compute the volume of your hippocampal prediction
    # label values are class ids: count the voxels of each class, then scale to mm^3
    volume_ant = round(float(np.sum(pred == 1)) * voxel_volume_mm3, 1)
    volume_post = round(float(np.sum(pred == 2)) * voxel_volume_mm3, 1)
    total_volume = round(float(np.sum((pred == 1) | (pred == 2))) * voxel_volume_mm3, 1)

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def quality_flags(inference, pred_vol, reference=None):
    """Simple automatic checks that tell the reader when to distrust a result.
    They catch some obvious failures only; passing them does not mean the mask is right."""
    flags = []
    mask = pred_vol > 0
    if inference["anterior"] == 0 or inference["posterior"] == 0:
        flags.append("A region is empty")
    if mask.any() and any(mask.take(i, axis=ax).any() for ax in range(3) for i in (0, -1)):
        flags.append("Mask touches the edge of the crop")
    limits = (reference or {}).get("training_volume_range_mm3")
    if limits and not limits[0] <= inference["total"] <= limits[1]:
        flags.append(f"Total outside the training label range {limits[0]:.0f}-{limits[1]:.0f} mm3")
    return flags

def slice_to_rgb(image_slice, mask_slice, low, high, size):
    """One axial slice, in radiological DICOM orientation, with a colour overlay, as PIL image"""
    grey = np.clip((image_slice.astype(float) - low) / max(high - low, 1e-6), 0, 1) * 255
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    for value, colour in COLOURS.items():
        rgb[mask_slice == value] = 0.55 * rgb[mask_slice == value] + 0.45 * np.array(colour)
    # undo load_dicom_volume_as_numpy_from_list's flip + transpose
    rgb = np.flip(rgb.transpose(1, 0, 2), axis=(0, 1)).astype(np.uint8)
    scale = min(size[0] / rgb.shape[1], size[1] / rgb.shape[0])
    return Image.fromarray(rgb, mode="RGB").resize(
        (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)), Image.NEAREST)

def create_report(inference, header, orig_vol, pred_vol, voxel_volume_mm3=1.0, reference=None):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype(FONT_PATH, size=40)
    main_font = ImageFont.truetype(FONT_PATH, size=22)
    small_font = ImageFont.truetype(FONT_PATH, size=18)

    # axial slices spread through the predicted structure (evenly through the crop if empty)
    occupied = np.where((pred_vol > 0).any(axis=(0, 1)))[0]
    if occupied.size == 0:
        occupied = np.arange(orig_vol.shape[2])
    slice_nums = [int(occupied[int(f * (occupied.size - 1))]) for f in (0.2, 0.5, 0.8)]

    # TASK: Create the report here and show information that you think would be relevant to
    # clinicians. A sample code is provided below, but feel free to use your creative
    # genius to make if shine. After all, the is the only part of all our machine learning
    # efforts that will be visible to the world. The usefulness of your computations will largely
    # depend on how you present them.
    white, grey, amber = (255, 255, 255), (170, 170, 170), (255, 190, 60)
    field = lambda name: str(header.get(name, "") or "not recorded")

    draw.text((20, 10), ALGORITHM_NAME, white, font=header_font)
    draw.text((330, 28), f"v{ALGORITHM_VERSION}  |  RESEARCH USE ONLY - NOT FOR DIAGNOSIS",
              amber, font=main_font)
    draw.multiline_text((20, 75),
                        f"Patient ID: {field('PatientID')}\n"
                        f"Accession Number: {field('AccessionNumber')}\n"
                        f"Study date: {field('StudyDate')}   Study: {field('StudyDescription')}\n"
                        f"Input series: {field('SeriesDescription')} ({field('Modality')}), "
                        f"{orig_vol.shape[2]} axial slices, voxel {voxel_volume_mm3:.3g} mm3",
                        white, font=main_font, spacing=8)

    draw.text((20, 215), "Predicted hippocampal volume (single crop)", white, font=main_font)
    rows = (("Anterior", inference["anterior"], COLOURS[1]),
            ("Posterior", inference["posterior"], COLOURS[2]),
            ("Total", inference["total"], white))
    for i, (name, value, colour) in enumerate(rows):
        y = 255 + 34 * i
        draw.rectangle([24, y + 4, 42, y + 22], fill=colour)
        draw.text((55, y), name, white, font=main_font)
        draw.text((200, y), f"{value:,.0f} mm3", white, font=main_font)

    flags = quality_flags(inference, pred_vol, reference)
    draw.text((440, 255), "Automatic checks:", white, font=main_font)
    for i, text in enumerate(flags or ["No flags raised (this does not confirm accuracy)"]):
        draw.text((440, 289 + 28 * i), "- " + text, amber if flags else grey, font=small_font)

    # image panels, shared grey-level window
    low, high = np.percentile(orig_vol, (1, 99))
    draw.text((20, 380), "Axial slices with predicted mask (patient right is on the left of each panel)",
              white, font=main_font)
    for i, k in enumerate(slice_nums):
        panel = slice_to_rgb(orig_vol[:, :, k], pred_vol[:, :, k], low, high, (310, 450))
        x = 20 + 325 * i + (310 - panel.width) // 2
        pimg.paste(panel, box=(x, 420))
        draw.text((20 + 325 * i, 875), f"axial slice {k + 1} of {orig_vol.shape[2]}", grey, font=small_font)

    performance = "Model performance: see the accompanying evaluation report."
    if reference and "mean_dice" in reference:
        performance = (f"Offline test (n={reference['n_test_volumes']} research crops): mean Dice "
                       f"{reference['mean_dice']:.3f}, total volume mean absolute error "
                       f"{reference['volume_mae_mm3']:.0f} mm3. Not clinically validated.")
    draw.multiline_text((20, 910), performance +
                        "\nVolume alone does not diagnose Alzheimer's disease. A qualified clinician must "
                        "review the mask\nand interpret the result. Model " +
                        (reference or {}).get("model_sha256", "unknown")[:12],
                        grey, font=small_font, spacing=5)

    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF
    # The code here is complete as it is unlikely that as a data scientist you will 
    # have to dive that deep into generating DICOMs. However, if you still want to understand
    # the subject, there are some suggestions below

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = copy.deepcopy(header)
    # pydicom >= 2 wants FileMetaDataset; the 1.4 release in the Udacity workspace has only Dataset
    out.file_meta = getattr(pydicom.dataset, "FileMetaDataset", pydicom.Dataset)()
    out.preamble = b"\0" * 128

    # attributes that describe the source MR slice and would be wrong for a report picture
    for keyword in ("PixelData", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
                    "ImagePositionPatient", "ImageOrientationPatient", "FrameOfReferenceUID",
                    "PositionReferenceIndicator", "SliceLocation", "WindowCenter", "WindowWidth",
                    "SmallestImagePixelValue", "LargestImagePixelValue", "RescaleSlope",
                    "RescaleIntercept"):
        if keyword in out:
            delattr(out, keyword)
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # STAND OUT SUGGESTION: 
    # If you want to understand better the generation of valid DICOM, remove everything below
    # and try writing your own DICOM generation code from scratch.
    # Refer to this part of the standard to see what are the requirements for the valid
    # Secondary Capture IOD: http://dicom.nema.org/medical/dicom/2019e/output/html/part03.html#sect_A.8
    # The Modules table (A.8-1) contains a list of modules with a notice which ones are mandatory (M)
    # and which ones are conditional (C) and which ones are user-optional (U)
    # Note that we are building an RGB image which would have three 8-bit samples per pixel
    # Also note that writing code that generates valid DICOM has a very calming effect
    # on mind and body :)

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date. Study date/time belong to the original study and are left unchanged
    # (a study must not report different values in different series).
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = header.get("StudyDate", "")
    out.StudyTime = header.get("StudyTime", "")
    out.SeriesDate = dt
    out.SeriesTime = tm
    out.ContentDate = dt
    out.ContentTime = tm
    out.InstanceCreationDate = dt
    out.InstanceCreationTime = tm

    out.ImagesInAcquisition = 1
    out.SeriesNumber = 9001
    out.InstanceNumber = 1

    # attributes required by the Secondary Capture IOD (type 1 / 2) that the crop lacks
    out.ConversionType = "WSD"           # created on a workstation
    out.ImageType = ["DERIVED", "SECONDARY"]
    out.Manufacturer = "Udacity student project"
    out.ManufacturerModelName = ALGORITHM_NAME
    out.SoftwareVersions = ALGORITHM_VERSION
    out.PatientOrientation = ""
    for keyword in ("PatientBirthDate", "PatientSex", "ReferringPhysicianName", "StudyID",
                    "AccessionNumber"):
        if keyword not in out:
            setattr(out, keyword, "")

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()
    out["PixelData"].VR = "OB"

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def incompatibilities(dcm):
    """Reasons why one instance cannot be processed by this model (empty list = acceptable)"""
    reasons = []
    if dcm.get("Modality") != "MR":
        reasons.append("modality is not MR")
    if "PixelData" not in dcm or "InstanceNumber" not in dcm:
        return reasons + ["missing pixel data or instance number"]
    if dcm.get("SamplesPerPixel", 1) != 1:
        reasons.append("not a greyscale image")
    # slices are sliced along left-right by the model, so AP and SI sizes must fit the patch
    if dcm.Rows > PATCH_SIZE:
        reasons.append(f"{dcm.Rows} rows exceed the {PATCH_SIZE} patch")
    spacing = [float(v) for v in dcm.get("PixelSpacing", [])]
    if len(spacing) != 2 or not np.allclose(spacing, 1.0, atol=0.05):
        reasons.append(f"in-plane spacing {spacing} is not 1 mm")
    orientation = [float(v) for v in dcm.get("ImageOrientationPatient", [])]
    if len(orientation) != 6 or not np.allclose(orientation, [1, 0, 0, 0, 1, 0], atol=0.05):
        reasons.append("slices are not axial in the expected orientation")
    return reasons

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    dicoms = []
    for folder, _, files in os.walk(path):
        for f in sorted(files):
            try:
                dicoms.append(pydicom.dcmread(os.path.join(folder, f)))
            except pydicom.errors.InvalidDicomError:
                print(f"Skipping non-DICOM file {f}")

    # TASK: create a series_for_inference variable that will contain a list of only 
    # those PyDicom objects that represent files that belong to the series that you 
    # will run inference on.
    # It is important to note that radiological modalities most often operate in terms
    # of studies, and it will most likely be on you to establish criteria for figuring 
    # out which one of the multiple series sent by the scanner is the one you need to feed to 
    # your algorithm. In our case it's rather easy - we have reached an agreement with 
    # people who configured the HippoCrop tool and they label the output of their tool in a 
    # certain way. Can you figure out which is that? 
    # Hint: inspect the metadata of HippoCrop series

    # HippoCrop labels its output with SeriesDescription "HippoCrop". The name alone is not
    # trusted: the geometry must also match what the model was trained on.
    series_for_inference = [d for d in dicoms
                            if str(d.get("SeriesDescription", "")).strip() == SERIES_DESCRIPTION_FOR_INFERENCE]
    if not series_for_inference:
        print(f"Error: no {SERIES_DESCRIPTION_FOR_INFERENCE} series found")
        return []

    problems = sorted({p for d in series_for_inference for p in incompatibilities(d)})
    if problems:
        print("Error: the HippoCrop series is not a supported input: " + "; ".join(problems))
        return []

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()
    return sp.returncode

    # Uncomment this if running under Windows
    # os.system(command)

def load_model_reference(model_path):
    """Checksum of the weights plus the measured test performance stored beside them"""
    with open(model_path, "rb") as fh:
        reference = {"model_sha256": hashlib.sha256(fh.read()).hexdigest()}
    card = os.path.join(os.path.dirname(model_path), "model_card.json")
    if os.path.exists(card):
        with open(card) as fh:
            info = json.load(fh)
        # only quote performance figures that belong to exactly these weights
        if info.get("model_sha256") == reference["model_sha256"]:
            reference.update(info)
    return reference

if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    parser = argparse.ArgumentParser(description="Run HippoVolume.AI on the latest routed study")
    parser.add_argument("routing_dir", help="folder with one sub-folder per routed study")
    parser.add_argument("--model", default=os.environ.get("HIPPO_MODEL") or os.path.normpath(
        os.path.join(SRC_DIR, "..", "models", "model.pth")))
    parser.add_argument("--report", default=os.path.normpath(os.path.join(SRC_DIR, "..", "outputs", "report.dcm")))
    parser.add_argument("--no-send", action="store_true", help="create the report but do not C-STORE it")
    parser.add_argument("--keep-study", action="store_true", help="never delete the routed study folder")
    args = parser.parse_args()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(args.routing_dir, d) for d in os.listdir(args.routing_dir) if
                os.path.isdir(os.path.join(args.routing_dir, d))]
    if not subdirs:
        sys.exit(f"No study folders found in {args.routing_dir}")

    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    # TASK: get_series_for_inference is not complete. Go and complete it
    series = get_series_for_inference(study_dir)
    if not series:
        sys.exit("Nothing to run inference on. The study folder was left untouched.")
    volume, header = load_dicom_volume_as_numpy_from_list(series)
    voxel_volume = get_voxel_volume_mm3(series)
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    # TASK: Use the UNetInferenceAgent class and model parameter file from the previous section
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=args.model,
        patch_size=PATCH_SIZE)
    reference = load_model_reference(args.model)

    # Run inference
    # TASK: single_volume_inference_unpadded takes a volume of arbitrary size 
    # and reshapes y and z dimensions to the patch size used by the model before 
    # running inference. Your job is to implement it.
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    # TASK: get_predicted_volumes is not complete. Go and complete it
    pred_volumes = get_predicted_volumes(pred_label, voxel_volume)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = args.report
    os.makedirs(os.path.dirname(os.path.abspath(report_save_path)), exist_ok=True)
    # TASK: create_report is not complete. Go and complete it. 
    # STAND OUT SUGGESTION: save_report_as_dcm has some suggestions if you want to expand your
    # knowledge of DICOM format
    report_img = create_report(pred_volumes, header, volume, pred_label, voxel_volume, reference)
    save_report_as_dcm(header, report_img, report_save_path)

    # Send report to our storage archive
    # TASK: Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    sent = False
    if args.no_send:
        print(f"--no-send: report saved to {report_save_path} and NOT sent to the PACS")
    else:
        status = os_command(f'storescu 127.0.0.1 4242 -v -aec HIPPOAI "{report_save_path}"')
        sent = status == 0
        if not sent:
            sys.exit(f"ERROR: storescu failed (exit status {status}). The report is at "
                     f"{report_save_path}; the routed study was kept.")

    # This line will remove the study dir if run as root user
    # Sleep to let our StoreSCP server process the report (remember - in our setup
    # the main archive is routing everyting that is sent to it, including our freshly generated
    # report) - we want to give it time to save before cleaning it up
    # The folder is removed only after a successful send, so a failed run loses nothing.
    if sent and not args.keep_study:
        time.sleep(2)
        shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
