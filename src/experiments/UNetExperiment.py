"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard, load_weights, save_weights
from utils.volume_stats import Dice3d, Jaccard3d, Sensitivity, Specificity
from networks.RecursiveUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:

        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        # an interrupted run is continued in its own folder (run_ml_pipeline.py --resume)
        self.resume_dir = getattr(config, "resume_dir", "")
        self.out_dir = self.resume_dir or os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create data loaders
        # TASK: SlicesDataset class is not complete. Go to the file and complete it. 
        # Note that we are using a 2D version of UNet here, which means that it will expect
        # batches of 2D slices.
        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)

        # we will access volumes directly for testing
        self.test_data = dataset[split["test"]]

        # Do we have CUDA available?
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure our model and other training implements
        # We will use a recursive UNet model from German Cancer Research Center, 
        # Division of Medical Image Computing. It is quite complicated and works 
        # very well on this task. Feel free to explore it or plug in your own model
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # We are using a standard cross-entropy loss since the model output is essentially
        # a tensor with softmax'd prediction of each pixel's probability of belonging 
        # to a certain class
        self.loss_function = torch.nn.CrossEntropyLoss()

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Set up Tensorboard. By default it saves data into runs folder. You need to launch
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")
        self.tensorboard_train_writer.add_text("config", str(vars(config)))

        self.best_val_loss = float("inf")
        self.history = []

        # Fixed validation slices (the middle-of-mask slice of 4 volumes) so that predictions
        # can be compared across training stages
        self.preview = []
        for volume in dataset[split["val"]][:4]:
            x = int(np.argmax((volume["seg"] > 0).sum(axis=(1, 2))))
            self.preview.append((volume["filename"], volume["image"][x], volume["seg"][x]))

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()
        epoch_losses = []

        # Loop over our minibatches
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # TASK: You have your data in batch variable. Put the slices as 4D Torch Tensors of 
            # shape [BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE] into variables data and target. 
            # Feed data to the model and feed target to the loss function
            # 
            data = batch["image"].to(self.device, dtype=torch.float)
            target = batch["seg"].to(self.device, dtype=torch.long)

            prediction = self.model(data)

            # We are also getting softmax'd version of prediction to output a probability map
            # so that we can see how the model converges to the solution
            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target[:, 0, :, :])

            # TASK: What does each dimension of variable prediction represent?
            # ANSWER: [batch size, number of classes (3: background, anterior, posterior),
            # slice height (64), slice width (64)]. Each value is an unnormalised score (logit)
            # for one class at one pixel.

            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")

                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                # You don't need to do anything with this function, but you are welcome to 
                # check it out if you want to see how images are logged to Tensorboard
                # or if you want to output additional debug data
                # The loss is logged on every 10th batch; the image grids (slow to render and
                # large on disk) only on every 100th.
                if (i % 100) == 0:
                    log_to_tensorboard(
                        self.tensorboard_train_writer,
                        loss,
                        data,
                        target,
                        prediction_softmax,
                        prediction,
                        counter)
                else:
                    self.tensorboard_train_writer.add_scalar("Loss", loss, counter)

            print(".", end='', flush=True)

        print("\nTraining complete")
        return float(np.mean(epoch_losses))

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        print(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                
                # TASK: Write validation code that will compute loss on a validation sample
                data = batch["image"].to(self.device, dtype=torch.float)
                target = batch["seg"].to(self.device, dtype=torch.long)
                prediction = self.model(data)
                prediction_softmax = F.softmax(prediction, dim=1)
                loss = self.loss_function(prediction, target[:, 0, :, :])

                if (i % 20) == 0:
                    print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax, 
            prediction,
            (self.epoch+1) * 100)
        self.log_preview((self.epoch+1) * 100)
        print(f"Validation complete")
        return float(np.mean(loss_list))

    def log_preview(self, counter):
        """Logs image / ground truth / prediction for the fixed validation slices"""
        figure, axes = plt.subplots(len(self.preview), 3, figsize=(7.5, 2.5 * len(self.preview)),
                                    squeeze=False)
        with torch.no_grad():
            for row, (name, image, seg) in enumerate(self.preview):
                data = torch.from_numpy(image.astype(np.float32))[None, None].to(self.device)
                pred = torch.argmax(self.model(data), dim=1)[0].cpu().numpy()
                for ax, mask, title in zip(axes[row], (None, seg, pred),
                                           (name.split(".")[0], "ground truth", "prediction")):
                    ax.imshow(image.T, cmap="gray", origin="lower")
                    if mask is not None:
                        ax.imshow(np.ma.masked_equal(mask, 0).T, cmap="autumn", vmin=1, vmax=2,
                                  alpha=0.5, origin="lower", interpolation="nearest")
                    ax.set_title(title, fontsize=9)
                    ax.axis("off")
        figure.suptitle(f"red = anterior, yellow = posterior (step {counter})", fontsize=9)
        figure.tight_layout()
        self.tensorboard_val_writer.add_figure("Validation preview", figure, global_step=counter)

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")

        save_weights(self.model.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)     # an interrupted run never leaves a truncated file

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(load_weights(model_path, self.device))
        else:
            raise Exception(f"Could not find path {model_path}")

    def save_training_state(self):
        """
        Saves everything needed to continue after the current epoch (about three times the
        size of model.pth because of the Adam moments). Failure to write is not fatal.
        """
        path = os.path.join(self.out_dir, "resume.pth")
        state = {"epoch": self.epoch, "model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                 "best_val_loss": self.best_val_loss, "best_epoch": self.best_epoch,
                 "history": self.history, "rng": torch.get_rng_state()}
        try:
            torch.save(state, path + ".tmp")
            os.replace(path + ".tmp", path)
        except (OSError, RuntimeError) as error:
            print(f"WARNING: could not write the resume file ({error})")
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")

    def load_training_state(self):
        """
        Restores the state written by save_training_state and returns the next epoch.
        The best weights so far are read back from model.pth in the same folder.
        """
        state = torch.load(os.path.join(self.out_dir, "resume.pth"), map_location=self.device)
        self.best_state = load_weights(os.path.join(self.out_dir, "model.pth"), "cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.best_val_loss, self.best_epoch = state["best_val_loss"], state["best_epoch"]
        self.history = state["history"]
        torch.set_rng_state(state["rng"].cpu())
        print(f"Resuming after epoch {state['epoch']}")
        return state["epoch"] + 1

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        print("Testing...")
        self.model.eval()

        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes, much like we will be doing it when we deploy the model in the 
        # clinical environment. 

        # TASK: Inference Agent is not complete. Go and finish it. Feel free to test the class
        # in a module of your own by running it against one of the data samples
        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []

        # for every in test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            # score on the original geometry (padding removed), as in deployment
            crop = tuple(slice(0, n) for n in x["original_shape"])
            pred_label, truth = pred_label[crop], x["seg"][crop]

            # We compute and report Dice and Jaccard similarity coefficients which 
            # assess how close our volumes are to each other

            # TASK: Dice3D and Jaccard3D functions are not implemented. 
            #  Complete the implementation as we discussed
            # in one of the course lessons, you can look up definition of Jaccard index 
            # on Wikipedia. If you completed it
            # correctly (and if you picked your train/val/test split right ;)),
            # your average Jaccard on your test set should be around 0.80

            dc = Dice3d(pred_label, truth)
            jc = Jaccard3d(pred_label, truth)
            dc_list.append(dc)
            jc_list.append(jc)

            # STAND-OUT SUGGESTION: By way of exercise, consider also outputting:
            # * Sensitivity and specificity (and explain semantic meaning in terms of 
            #   under/over segmenting)
            # * Dice-per-slice and render combined slices with lowest and highest DpS
            # * Dice per class (anterior/posterior)

            stats = {
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc,
                "sensitivity": Sensitivity(pred_label, truth),
                "specificity": Specificity(pred_label, truth),
                }
            for value, region in ((1, "anterior"), (2, "posterior")):
                stats[f"dice_{region}"] = Dice3d(pred_label == value, truth == value)
                stats[f"jaccard_{region}"] = Jaccard3d(pred_label == value, truth == value)
            voxel = x["voxel_volume_mm3"]
            for values, region in (((1,), "anterior"), ((2,), "posterior"), ((1, 2), "total")):
                predicted = float(np.isin(pred_label, values).sum() * voxel)
                reference = float(np.isin(truth, values).sum() * voxel)
                stats[f"volume_{region}_pred_mm3"] = predicted
                stats[f"volume_{region}_true_mm3"] = reference
            out_dict["volume_stats"].append(stats)
            print(f"{x['filename']} Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

        rows = out_dict["volume_stats"]
        column = lambda key: np.array([r[key] for r in rows], dtype=float)
        overall = {
            "mean_dice": float(np.mean(dc_list)),
            "mean_jaccard": float(np.mean(jc_list)),
            "std_dice": float(np.std(dc_list, ddof=1)) if len(rows) > 1 else 0.0,
            "median_dice": float(np.median(dc_list)),
            "min_dice": float(np.min(dc_list)),
            "max_dice": float(np.max(dc_list)),
            "n_test_volumes": len(rows)}
        for key in ("dice_anterior", "dice_posterior", "jaccard_anterior", "jaccard_posterior",
                    "sensitivity", "specificity"):
            overall[f"mean_{key}"] = float(np.mean(column(key)))
        for region in ("anterior", "posterior", "total"):
            predicted, reference = column(f"volume_{region}_pred_mm3"), column(f"volume_{region}_true_mm3")
            error = predicted - reference
            overall[f"volume_{region}"] = {
                "mean_absolute_error_mm3": float(np.mean(np.abs(error))),
                "mean_absolute_percentage_error": float(100 * np.mean(np.abs(error) / reference)),
                "bias_mm3": float(np.mean(error)),
                "limits_of_agreement_mm3": [
                    float(np.mean(error) - 1.96 * np.std(error, ddof=1)),
                    float(np.mean(error) + 1.96 * np.std(error, ddof=1))] if len(rows) > 1 else None,
                "pearson_r": float(np.corrcoef(predicted, reference)[0, 1]) if len(rows) > 1 else None}
        out_dict["overall"] = overall

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        first_epoch = self.load_training_state() if self.resume_dir else 0
        for self.epoch in range(first_epoch, self.n_epochs):
            started = time.time()
            train_loss = self.train()
            val_loss = self.validate()

            # keep the parameters of the epoch with the lowest validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_epoch = val_loss, self.epoch
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                try:
                    self.save_model_parameters()
                except (OSError, RuntimeError) as error:     # e.g. disk full: keep training
                    print(f"WARNING: could not write checkpoint ({error}); best weights kept in memory")
            self.history.append({"epoch": self.epoch, "train_loss": train_loss, "val_loss": val_loss,
                                 "learning_rate": self.optimizer.param_groups[0]["lr"],
                                 "seconds": round(time.time() - started, 1)})
            print(f"Epoch {self.epoch} summary: {self.history[-1]}", flush=True)
            self.save_training_state()

        # test (and deploy) the best epoch
        self.model.load_state_dict(self.best_state)
        self.save_model_parameters()
        resume_file = os.path.join(self.out_dir, "resume.pth")
        if os.path.exists(resume_file):
            os.remove(resume_file)      # not needed once training is complete
        self.tensorboard_train_writer.close()
        self.tensorboard_val_writer.close()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
