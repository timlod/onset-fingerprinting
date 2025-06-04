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
from onset_fingerprinting.model import CNN, RNN, CNNRNN

torch.set_float32_matmul_precision("medium")

data_dir = Path("../data/location/Recordings3")
epochs = 3000
w = 320
channels = 3
outdim = 2

test = np.load(data_dir / "data.npy")
lugdata = np.load(data_dir / "lugdata.npy")
test_onsets = np.load(data_dir / "onsets.npy")
lugonsets = np.load(data_dir / "lugonsets.npy")
test_sp = np.load(data_dir / "sp.npy")
lugsp = np.load(data_dir / "lugsp.npy")
pre_samp = -16
# This is quite slow, so it might be better to precompute some stretches and
# shift those
sfe = data.StretchFrameExtractor(w, 0, 0.03)
dataset = data.MCPOSD(lugdata, lugonsets, lugsp, w, pre_samp, 32, 20, cc=False)
train = dataset
# train, val = dataset.split()
test_dataset = data.MCPOSD(test, test_onsets, test_sp, w, cc=False)
val, test = test_dataset.split(0.1)
tdl = DataLoader(train, batch_size=None)
vdl = DataLoader(val, batch_size=None)
testdl = DataLoader(test_dataset, batch_size=None)


def objective(trial: optuna.trial.Trial) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layer_sizes = [
        trial.suggest_int("out_channels_l{}".format(i), 4, 64, log=True)
        for i in range(n_layers)
    ]
    n_hidden = trial.suggest_int("n_hidden", 16, 128, step=2)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.3, 0.8)
    kernel_size = trial.suggest_int("kernel_size", 2, 8)
    # lossfun = trial.suggest_categorical("lossfun", [F.l1_loss, F.mse_loss])
    # batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    # pool = trial.suggest_categorical("pool", [True, False])
    # padding = trial.suggest_int("padding", 0, 1)
    # dilation = trial.suggest_int("dilation", 1, 2)
    # max_shift = trial.suggest_int("max_shift", 8, 64, step=8)

    # layer_sizes = [16, 64, 32]
    # kernel_size = 3
    dilation = 1
    padding = 1
    # dropout = 0.5
    batch_norm = True
    pool = False
    lossfun = F.mse_loss

    model = CNNRNN(
        input_size=w,
        # with cc
        # input_size=2 * w - 1,
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
        lr=0.0001,
        activation=nn.SiLU,
        groups=1,
        n_hidden=n_hidden,
        n_rnn_layers=n_rnn_layers,
    )
    # hs = trial.suggest_int("hs", 4, 64)
    # hs = 128
    # nl = trial.suggest_int("nl", 1, 4)
    # nl = 2
    # dropout = trial.suggest_float("dropout", 0.1, 0.5)
    # dropout = 0.8
    # rnn_type = trial.suggest_categorical("rnn_type", ["LSTM", "GRU", "RNN"])
    # rnn_type = "GRU"

    # model = RNN(
    #     w,
    #     outdim,
    #     channels,
    #     hs,
    #     nl,
    #     dropout,
    #     rnn_type=rnn_type,
    #     loss=F.mse_loss,
    #     lr=0.0001,
    # )

    trainer = L.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=3000,
        max_steps=-1,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[
            # PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            EarlyStopping(monitor="val_loss", mode="min", patience=300),
            # Stochasticweightaveraging(
            #     swa_lrs=1e-4, swa_epoch_start=100, annealing_epochs=10
            # ),
        ],
        # min_epochs=1000,
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
        n_hidden=n_hidden,
        n_rnn_layers=n_rnn_layers,
        # hs=hs,
        # nl=nl,
        # dropout=dropout,
        # rnn_type=rnn_type,
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
    study.optimize(objective, n_trials=100, catch=[RuntimeError])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
