"""Tests for the DICOM stage. Test studies are used when present, synthetic data otherwise.
None of these tests says anything about model accuracy."""
import glob
import os
import sys

import numpy as np
import pydicom
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import inference_dcm as dcm_app        # noqa: E402

STUDIES = sorted(glob.glob(os.path.join(ROOT, "data", "TestVolumes", "Study*")))
needs_data = pytest.mark.skipif(not STUDIES, reason="test DICOM studies not downloaded")


def test_predicted_volumes_count_voxels_and_use_spacing():
    pred = np.zeros((4, 4, 4), np.uint8)
    pred[0, 0, :3] = 1
    pred[1, 0, :2] = 2
    assert dcm_app.get_predicted_volumes(pred) == {"anterior": 3, "posterior": 2, "total": 5}
    assert dcm_app.get_predicted_volumes(pred, 2.5) == {"anterior": 7.5, "posterior": 5.0, "total": 12.5}


@needs_data
@pytest.mark.parametrize("study", STUDIES)
def test_series_selection_and_geometry(study):
    series = dcm_app.get_series_for_inference(study)
    assert series and {d.SeriesDescription for d in series} == {"HippoCrop"}
    volume, header = dcm_app.load_dicom_volume_as_numpy_from_list(series)
    first = min(series, key=lambda d: d.InstanceNumber)
    assert volume.shape == (first.Columns, first.Rows, len(series))
    # axis 0 -> patient right, axis 1 -> anterior: both opposite to DICOM column/row order
    assert volume[0, 0, 0] == first.pixel_array[-1, -1]
    assert header.PixelData is None
    assert dcm_app.get_voxel_volume_mm3(series) == pytest.approx(1.0)


@needs_data
def test_rejects_incompatible_and_missing_series(tmp_path):
    source = dcm_app.get_series_for_inference(STUDIES[0])
    assert dcm_app.get_series_for_inference(str(tmp_path)) == []
    for i, d in enumerate(source):
        d.PixelSpacing = [0.5, 0.5]
        d.save_as(tmp_path / f"{i}.dcm")
    assert dcm_app.get_series_for_inference(str(tmp_path)) == []


@needs_data
def test_missing_slice_is_detected():
    series = sorted(dcm_app.get_series_for_inference(STUDIES[0]), key=lambda d: d.InstanceNumber)
    with pytest.raises(ValueError):
        dcm_app.get_voxel_volume_mm3(series[:5] + series[6:])


@needs_data
def test_report_is_a_readable_secondary_capture(tmp_path):
    series = dcm_app.get_series_for_inference(STUDIES[0])
    volume, header = dcm_app.load_dicom_volume_as_numpy_from_list(series)
    pred = np.zeros(volume.shape, np.uint8)
    pred[10:20, 10:20, 10:20] = 1
    image = dcm_app.create_report(dcm_app.get_predicted_volumes(pred), header, volume, pred)
    path = str(tmp_path / "report.dcm")
    dcm_app.save_report_as_dcm(header, image, path)

    out = pydicom.dcmread(path)          # strict read: preamble and file meta must be present
    source = series[0]
    assert out.SOPClassUID == out.file_meta.MediaStorageSOPClassUID == "1.2.840.10008.5.1.4.1.1.7"
    assert out.SOPInstanceUID == out.file_meta.MediaStorageSOPInstanceUID
    assert out.StudyInstanceUID == source.StudyInstanceUID
    assert out.StudyDate == source.StudyDate and out.PatientID == source.PatientID
    assert out.SeriesInstanceUID != source.SeriesInstanceUID
    assert out.SOPInstanceUID not in {d.SOPInstanceUID for d in series}
    assert out.ConversionType == "WSD" and "PixelSpacing" not in out
    assert np.array_equal(out.pixel_array, np.asarray(image))
