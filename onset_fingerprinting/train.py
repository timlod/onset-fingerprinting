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
from onset_fingerprinting import model

torch.set_float32_matmul_precision("medium")

data_dir = Path("../data/location/Recordings3")
epochs = 7500
w = 256
pre_samp = 0
channels = [0, 1, 2]
channels = None
# This is quite slow, so it might be better to precompute some stretches and
# shift those
# sfe = data.StretchFrameExtractor(w, 0, 0.03)
dataset = data.MCPOSD.from_file(
    data_dir / "Setup 1", "combined0", w, pre_samp, 16, 16, channels=channels
)
extra_pos = [
    488,
    489,
    490,
    491,
    492,
    493,
    494,
    495,
    680,
    681,
    682,
    683,
    684,
    685,
    686,
    687,
    712,
    713,
    714,
    715,
    716,
    717,
    718,
    719,
    904,
    905,
    906,
    907,
    908,
    909,
    910,
    911,
]
pos = np.r_[:72, extra_pos]
train = data.MCPOSD(
    dataset.data,
    dataset.frame_extractor.onsets[pos],
    dataset[0][1][pos],
    w,
    pre_samp,
    16,
    32,
    channels=channels,
)
# train = data.MCPOSD.from_xy(dataset[0][0][::8], dataset[0][1][::8])
test_dataset = data.MCPOSD.from_file(
    data_dir / "Setup 1", "combined0", w, 0, 0, 1, channels=channels
)
# test_dataset = data.MCPOSD(test, test_onsets, test_sp, w)
val, test = test_dataset.split(0.1)
tdl = DataLoader(train, batch_size=None)
vdl = DataLoader(val, batch_size=None)
testdl = DataLoader(test_dataset, batch_size=None)
channels = dataset[0][0].shape[1]
outdim = dataset[0][1].shape[1]


def objective(trial: optuna.trial.Trial) -> float:
    # n_layers = trial.suggest_int("n_layers", 1, 3)
    # layer_sizes = [
    #     trial.suggest_int("out_channels_l{}".format(i), 2, 128, log=True)
    #     for i in range(n_layers)
    # ]
    # dropout = trial.suggest_float("dropout", 0.0, 0.1)
    # kernel_size = trial.suggest_int("kernel_size", 3, 5)

    m = model.LCCCNN(
        w,
        outdim,
        channels,
        layer_sizes=[1] * 9,
        # kernel_sizes=11,
        kernel_sizes=[15, 15, 11, 7],
        # kernel_sizes=[13] * 4,
        strides=[2, 2] + 10 * [1],
        dropout_rate=0.0,
        batch_norm=True,
        loss=F.huber_loss,
        lr=0.0001,
        group=False,
        pool=False,
    )

    trainer = L.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=epochs,
        max_steps=-1,
        accelerator="auto",
        devices=1,
        callbacks=[
            # PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            EarlyStopping(monitor="val_loss", mode="min", patience=500),
            # StochasticWeightAveraging(swa_lrs=1e-3),
            # model.GradProbe(),
        ],
        min_epochs=1000,
        # gradient_clip_val=1.5,
        # detect_anomaly=True,
    )

    # tuner = Tuner(trainer)
    # tuner.lr_find(m, train_dataloaders=tdl, val_dataloaders=vdl, max_lr=0.1)
    trainer.fit(m, train_dataloaders=tdl, val_dataloaders=vdl)
    trainer.test(m, testdl)

    return trainer.callback_metrics["hp_metric"].item()


# Input paired CCs, output locations
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=1, catch=[RuntimeError])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
