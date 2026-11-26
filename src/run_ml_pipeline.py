"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import random
import shutil
import argparse

import numpy as np
import torch

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        # curated dataset written by the EDA notebook (override with HIPPO_ROOT)
        self.root_dir = os.environ.get("HIPPO_ROOT", os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "curated")))
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = os.environ.get("HIPPO_RESULTS", os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "experiments")))
        self.seed = 42

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    parser = argparse.ArgumentParser(description="Train and test the hippocampus U-Net")
    parser.add_argument("--epochs", type=int, default=c.n_epochs)
    parser.add_argument("--max-volumes", type=int, default=0, help="smoke test on a small subset")
    parser.add_argument("--resume", default="", metavar="EXPERIMENT_DIR",
                        help="continue an interrupted run from its resume.pth")
    args = parser.parse_args()
    c.n_epochs = args.epochs
    c.resume_dir = os.path.abspath(args.resume) if args.resume else ""

    # a checkpoint plus resume file need about 1 GB while being rewritten
    os.makedirs(c.test_results_dir, exist_ok=True)
    free_gb = shutil.disk_usage(c.test_results_dir).free / 1e9
    if free_gb < 2:
        raise SystemExit(f"Only {free_gb:.1f} GB free on the output disk; free at least 2 GB first")

    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    if args.max_volumes:
        data = data[:args.max_volumes]
    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # 70 / 15 / 15 split made by whole volume (never by slice), stratified by hippocampal
    # volume: sort by size, then give every consecutive block of 20 volumes 14/3/3 members
    # in a seeded random order. NOTE: the files carry no subject ids, so two volumes from one
    # person could still fall in different partitions.
    rng = np.random.RandomState(c.seed)
    by_size = sorted(keys, key=lambda k: (data[k]["seg"] > 0).sum() * data[k]["voxel_volume_mm3"])
    split = {"train": [], "val": [], "test": []}
    for start in range(0, len(by_size), 20):
        block = rng.permutation(by_size[start:start + 20])
        n_val = n_test = int(round(0.15 * len(block)))
        split["val"] += block[:n_val].tolist()
        split["test"] += block[n_val:n_val + n_test].tolist()
        split["train"] += block[n_val + n_test:].tolist()
    split = {name: np.array(sorted(ids), dtype=int) for name, ids in split.items()}
    assert sum(len(v) for v in split.values()) == len(set(np.concatenate(list(split.values())))) == len(data)
    print({name: len(ids) for name, ids in split.items()})

    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)
    results_json["split"] = {name: [data[i]["filename"] for i in ids] for name, ids in split.items()}
    results_json["history"] = exp.history
    results_json["best_epoch"] = exp.best_epoch
    results_json["device"] = str(exp.device)
    results_json["torch_version"] = torch.__version__

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

