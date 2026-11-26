"""Unit tests on synthetic arrays. They check code correctness, not model performance."""
import os
import sys

import numpy as np
import pytest
import torch

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from data_prep.SlicesDataset import SlicesDataset                # noqa: E402
from inference.UNetInferenceAgent import UNetInferenceAgent      # noqa: E402
from networks.RecursiveUNet import UNet                          # noqa: E402
from utils.utils import med_reshape, normalize_volume            # noqa: E402
from utils.volume_stats import Dice3d, Jaccard3d, Sensitivity, Specificity  # noqa: E402


def masks():
    a = np.zeros((4, 4, 4), int)
    b = a.copy()
    a[0, 0, :4] = 1          # 4 voxels
    b[0, 0, 2:4] = 2         # 2 voxels shared
    b[1, 0, :4] = 1          # 4 voxels only in b -> |b| = 6
    return a, b


def test_overlap_metrics_known_values():
    a, b = masks()
    assert Dice3d(a, b) == pytest.approx(2 * 2 / (4 + 6))
    assert Jaccard3d(a, b) == pytest.approx(2 / 8)
    assert Sensitivity(a, b) == pytest.approx(2 / 6)
    assert Specificity(a, b) == pytest.approx((64 - 8) / (64 - 6))


def test_overlap_metrics_edge_cases():
    empty, full = np.zeros((2, 2, 2)), np.ones((2, 2, 2))
    assert Dice3d(empty, empty) == Jaccard3d(empty, empty) == 1.0
    assert Dice3d(empty, full) == Jaccard3d(empty, full) == 0.0
    assert Dice3d(full, full) == 1.0
    assert np.isnan(Sensitivity(full, empty))
    for metric in (Dice3d, Jaccard3d):
        with pytest.raises(Exception):
            metric(empty, np.zeros((2, 2, 3)))
        with pytest.raises(Exception):
            metric(np.zeros((2, 2)), np.zeros((2, 2)))


def test_med_reshape_pads_crops_and_keeps_labels():
    label = np.random.randint(0, 3, (3, 5, 7))
    out = med_reshape(label, (3, 8, 8))
    assert np.array_equal(out[:, :5, :7], label)
    assert out.sum() == label.sum() and set(np.unique(out)) <= {0, 1, 2}
    assert np.array_equal(med_reshape(label, (3, 4, 4)), label[:, :4, :4])


def test_normalize_volume():
    x = np.array([[[10.0, 20.0, 30.0]]])
    assert np.allclose(normalize_volume(x), [[[0, .5, 1]]])
    assert np.all(normalize_volume(np.full((2, 2, 2), 7.0)) == 0)
    assert np.allclose(normalize_volume(x), normalize_volume(normalize_volume(x)))
    with pytest.raises(ValueError):
        normalize_volume(np.array([[[np.nan]]]))


def test_slices_dataset_shapes_and_dtypes():
    volume = {"image": np.random.rand(5, 64, 64), "seg": np.random.randint(0, 3, (5, 64, 64))}
    dataset = SlicesDataset([volume, volume])
    sample = dataset[7]
    assert len(dataset) == 10
    assert sample["image"].shape == sample["seg"].shape == (1, 64, 64)
    assert sample["image"].dtype == torch.float32 and sample["seg"].dtype == torch.int64
    assert np.array_equal(sample["seg"][0].numpy(), volume["seg"][2])


@pytest.fixture(scope="module")
def small_agent():
    torch.manual_seed(0)
    return UNetInferenceAgent(model=UNet(num_classes=3, initial_filter_size=4))


def test_inference_restores_geometry(small_agent):
    volume = np.random.rand(3, 35, 51) * 1000
    pred = small_agent.single_volume_inference_unpadded(volume)
    assert pred.shape == volume.shape and pred.dtype == np.uint8
    assert set(np.unique(pred)) <= {0, 1, 2}
    padded = med_reshape(normalize_volume(volume), (3, 64, 64))
    assert np.array_equal(pred, small_agent.single_volume_inference(padded)[:, :35, :51])


def test_inference_rejects_unsupported_input(small_agent):
    for bad in (np.zeros((3, 512, 512)), np.zeros((64, 64))):
        with pytest.raises(ValueError):
            small_agent.single_volume_inference_unpadded(bad)


def test_unet_output_and_checkpoint_roundtrip(tmp_path):
    model = UNet(num_classes=3, initial_filter_size=4)
    x = torch.rand(2, 1, 64, 64)
    assert model(x).shape == (2, 3, 64, 64)
    path = tmp_path / "m.pth"
    torch.save(model.state_dict(), path)
    again = UNet(num_classes=3, initial_filter_size=4)
    again.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(model.eval()(x), again.eval()(x))


def test_interrupted_training_resumes(tmp_path, monkeypatch):
    import types
    import experiments.UNetExperiment as module

    monkeypatch.setattr(module, "UNet", lambda num_classes: UNet(num_classes=num_classes, initial_filter_size=2))
    monkeypatch.chdir(tmp_path)         # TensorBoard writes ./runs
    volume = {"image": np.random.rand(2, 64, 64).astype(np.float32),
              "seg": np.random.randint(0, 3, (2, 64, 64)), "filename": "synthetic",
              "original_shape": (2, 64, 64), "voxel_volume_mm3": 1.0}
    data = np.array([volume, volume, volume])
    split = {"train": np.array([0]), "val": np.array([1]), "test": np.array([2])}
    config = types.SimpleNamespace(name="t", n_epochs=1, learning_rate=1e-3, batch_size=2,
                                   test_results_dir=str(tmp_path), resume_dir="")
    first = module.UNetExperiment(config, split, data)
    # simulate a crash after epoch 0 by not letting the finished run delete its resume file
    real_remove = module.os.remove
    monkeypatch.setattr(module.os, "remove", lambda p: None)
    first.run()
    monkeypatch.setattr(module.os, "remove", real_remove)
    assert os.path.exists(os.path.join(first.out_dir, "resume.pth"))

    config.n_epochs, config.resume_dir = 2, first.out_dir
    second = module.UNetExperiment(config, split, data)
    second.run()
    assert [h["epoch"] for h in second.history] == [0, 1]
    assert not os.path.exists(os.path.join(first.out_dir, "resume.pth"))
