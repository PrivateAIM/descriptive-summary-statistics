import numpy as np
import nibabel as nib
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset

from data_report.statistical_analysis.local.imaging_data_analysis import (
    distribution_by_anatomical_region,
    distribution_by_modality,
    file_size_statistics,
    resolution_statistics,
)


def make_dicom(path, rows=64, columns=64, modality="CT", body_part="CHEST", n_frames=None):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.PatientID = "TESTPAT1"
    ds.Modality = modality
    if body_part is not None:
        ds.BodyPartExamined = body_part
    ds.Rows = rows
    ds.Columns = columns
    if n_frames is not None:
        ds.NumberOfFrames = n_frames
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    frames = n_frames or 1
    ds.PixelData = np.zeros((frames, rows, columns), dtype=np.uint8).tobytes()
    ds.save_as(str(path), enforce_file_format=True, implicit_vr=True, little_endian=True)


def make_nifti(path, shape=(10, 12, 5)):
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine=np.eye(4))
    nib.save(img, str(path))


def make_png(path, size=(20, 30)):
    Image.new("L", size).save(path)


def test_file_size_statistics(tmp_path):
    make_png(tmp_path / "a.png", size=(10, 10))
    make_png(tmp_path / "b.png", size=(10, 10))
    (tmp_path / "ignored.txt").write_text("not an image")

    result = file_size_statistics(tmp_path)
    assert result["n_files"] == 2
    assert result["min_size"] <= result["max_size"]


def test_file_size_statistics_empty_dir_returns_empty_dict(tmp_path):
    assert file_size_statistics(tmp_path) == {}


def test_resolution_statistics_dicom_2d(tmp_path):
    make_dicom(tmp_path / "scan1.dcm", rows=64, columns=128)
    make_dicom(tmp_path / "scan2.dcm", rows=32, columns=64)

    result = resolution_statistics(tmp_path)
    assert result["n_images"] == 2
    assert result["min_width"] == 64
    assert result["max_width"] == 128
    assert result["min_height"] == 32
    assert result["max_height"] == 64
    assert "min_depth" not in result


def test_resolution_statistics_dicom_3d_volume(tmp_path):
    make_dicom(tmp_path / "volume.dcm", rows=64, columns=64, n_frames=20)

    result = resolution_statistics(tmp_path)
    assert result["min_depth"] == 20
    assert result["max_depth"] == 20


def test_resolution_statistics_nifti(tmp_path):
    make_nifti(tmp_path / "brain.nii.gz", shape=(10, 12, 5))

    result = resolution_statistics(tmp_path)
    assert result["min_width"] == 10
    assert result["min_height"] == 12
    assert result["min_depth"] == 5


def test_resolution_statistics_raster(tmp_path):
    make_png(tmp_path / "xray.png", size=(20, 30))

    result = resolution_statistics(tmp_path)
    assert result["min_width"] == 20
    assert result["min_height"] == 30
    assert "min_depth" not in result


def test_resolution_statistics_mixed_formats(tmp_path):
    make_dicom(tmp_path / "scan.dcm", rows=64, columns=64)
    make_nifti(tmp_path / "brain.nii.gz", shape=(10, 12, 5))
    make_png(tmp_path / "xray.png", size=(20, 30))

    result = resolution_statistics(tmp_path)
    assert result["n_images"] == 3


def test_resolution_statistics_empty_dir_returns_empty_dict(tmp_path):
    assert resolution_statistics(tmp_path) == {}


def test_distribution_by_modality(tmp_path):
    make_dicom(tmp_path / "ct1.dcm", modality="CT")
    make_dicom(tmp_path / "ct2.dcm", modality="CT")
    make_dicom(tmp_path / "mr1.dcm", modality="MR")
    make_png(tmp_path / "xray.png")  # no modality tag, ignored

    assert distribution_by_modality(tmp_path) == {"CT": 2, "MR": 1}


def test_distribution_by_modality_empty_dir_returns_empty_dict(tmp_path):
    assert distribution_by_modality(tmp_path) == {}


def test_distribution_by_anatomical_region_not_degenerate(tmp_path):
    make_dicom(tmp_path / "scan1.dcm", body_part="CHEST")
    make_dicom(tmp_path / "scan2.dcm", body_part="ABDOMEN")

    result = distribution_by_anatomical_region(tmp_path)
    assert result["counts"] == {"CHEST": 1, "ABDOMEN": 1}
    assert result["is_degenerate"] is False
    assert "reliability_note" in result


def test_distribution_by_anatomical_region_degenerate(tmp_path):
    make_dicom(tmp_path / "scan1.dcm", body_part="CHEST")
    make_dicom(tmp_path / "scan2.dcm", body_part="CHEST")

    result = distribution_by_anatomical_region(tmp_path)
    assert result["counts"] == {"CHEST": 2}
    assert result["is_degenerate"] is True


def test_distribution_by_anatomical_region_missing_tag_is_unknown(tmp_path):
    make_dicom(tmp_path / "scan1.dcm", body_part=None)

    result = distribution_by_anatomical_region(tmp_path)
    assert result["counts"] == {"Unknown": 1}


def test_distribution_by_anatomical_region_empty_dir_returns_empty_dict(tmp_path):
    assert distribution_by_anatomical_region(tmp_path) == {}
