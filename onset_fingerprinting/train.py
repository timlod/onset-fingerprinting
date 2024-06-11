from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from optuna.integration import PyTorchLightningPruningCallback
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from onset_fingerprinting import data
from onset_fingerprinting.model import CNN

torch.set_float32_matmul_precision("medium")

data_dir = Path("../data/location/Recordings3")
epochs = 5000
w = 320
channels = 3
outdim = 2

test = np.load(data_dir / "data.npy")
lugdata = np.load(data_dir / "lugdata.npy")
test_onsets = np.load(data_dir / "onsets.npy")
lugonsets = np.load(data_dir / "lugonsets.npy")
test_sp = np.load(data_dir / "sp.npy")
lugsp = np.load(data_dir / "lugsp.npy")
pre_samp = 16
# This is quite slow, so it might be better to precompute some stretches and
# shift those
sfe = data.StretchFrameExtractor(w, 0, 0.03)
dataset = data.MCPOSD(lugdata, lugonsets, lugsp, w, pre_samp, 32, 10)
train = dataset
# train, val = dataset.split()
test_dataset = data.MCPOSD(test, test_onsets, test_sp, w)
val, test = test_dataset.split(0.1)
tdl = DataLoader(train, batch_size=None)
vdl = DataLoader(val, batch_size=None)
testdl = DataLoader(test_dataset, batch_size=None)


def objective(trial: optuna.trial.Trial) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layer_sizes = [
        trial.suggest_int("out_channels_l{}".format(i), 2, 128, log=True)
        for i in range(n_layers)
    ]
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    kernel_size = trial.suggest_int("kernel_size", 2, 11)
    lossfun = trial.suggest_categorical("lossfun", [F.l1_loss, F.mse_loss])
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    pool = trial.suggest_categorical("pool", [True, False])
    padding = trial.suggest_int("padding", 0, 1)
    dilation = trial.suggest_int("dilation", 1, 2)

    model = CNN(
        input_size=w,
        output_size=outdim,
        channels=channels,
        layer_sizes=layer_sizes,
        kernel_size=kernel_size,
        dropout_rate=dropout,
        loss=lossfun,
        batch_norm=batch_norm,
        pool=pool,
        padding=padding,
        dilation=dilation,
        lr=0.001,
    )

    trainer = L.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=10000,
        max_steps=-1,
        accelerator="auto",
        devices=1,
        callbacks=[
            # PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            EarlyStopping(monitor="val_loss", mode="min", patience=500),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        min_epochs=1000,
    )
    hyperparameters = dict(
        n_layers=n_layers,
        layer_sizes=layer_sizes,
        kernel_size=kernel_size,
        dropout=dropout,
        loss=lossfun,
        batch_norm=batch_norm,
        pool=pool,
        padding=padding,
        dilation=dilation,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    # tuner = Tuner(trainer)
    # tuner.lr_find(
    #     model, train_dataloaders=tdl, val_dataloaders=vdl, max_lr=0.1
    # )
    trainer.fit(model, train_dataloaders=tdl, val_dataloaders=vdl)
    trainer.test(model, testdl)

    return trainer.callback_metrics["hp_metric"].item()


# Input paired CCs, output locations
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=3, catch=[RuntimeError])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
