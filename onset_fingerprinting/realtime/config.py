# Global configuration for loopmate
import json
from math import ceil
from pathlib import Path

from numpy import array, ndarray

WRITE_DIR = Path(__file__).parent.parent.parent / "data" / "demo"
VST_DIR = Path.home() / ".vst3"
fx = str(VST_DIR / "HY-Filter4 free.vst3")

# Global sample rate for all audio being played/recorded - if loading different
# SR files, convert them
SR = 96000
# Channels to record - they start at 0! TODO: Currently, this is not using
# direct specification of what to record within the host API, meaning that if
# you want to record channel 31 and 32, loopmate will always record 32
# channels, and slice the last two. That will mean a significant decrease in
# efficiency. For now, only one or two channels will work correctly, and they
# need to be the first two

# TODO: Fix to handle different input and output channels
CHANNELS = array([0, 1, 2])
N_CHANNELS = max(CHANNELS) + 1
CHANS_IN_OUT = N_CHANNELS, 2
# TODO: allow configuration of this to not necessarily always record everything
RECORD_CHANNELS = CHANNELS
DEVICE = [19, 22]
# Change this to your other device used for headphone output.
HEADPHONE_DEVICE = "default"
# Desired latency of audio interface, in ms
LATENCY = 0.001
# Blocksize to use in processing, the lower the higher the CPU usage, and lower
# the latency
BLOCKSIZE = 128
# Length (in ms) of blending window, e.g. used to remove pops in muting, or
# applying transformations to audio
BLEND_LENGTH = 0.05
QUANTIZE_MS = 0.2
# Output delay from speaker sound travel
AIR_DELAY = 0.0
# Maximum recording length (in seconds). Will constantly keep a buffer of sr *
# this many samples to query backwards from.
MAX_RECORDING_LENGTH = 60
# MIDI (output) port to use as MIDI input
MIDI_PORT = 0
MIDI_CHANNEL = 0

# STFT config
## Think about defining this in ms, such that it gives the same temporal
## resolution regardless of sampling rate
N_FFT = 2048
HOP_LENGTH = BLOCKSIZE
TG_WIN_LENGTH = 1024
TG_PAD = 2 * TG_WIN_LENGTH - 1


REC_N = MAX_RECORDING_LENGTH * SR
BLEND_SAMPLES = round(SR * BLEND_LENGTH)


def save_setup(sensor_locations: ndarray | list, medium, c, p: Path | str):
    if isinstance(sensor_locations, ndarray):
        sensor_locations = sensor_locations.tolist()
    with open(p, "w") as f:
        json.dump(
            {"sensor_locations": sensor_locations, "medium": medium, "c": c}, f
        )


def load_setup(p: Path, c=None):
    with open(p) as f:
        conf = json.load(f)
    conf["sensor_locations"] = array(conf["sensor_locations"])
    if c is not None:
        conf["c"] = c
    return conf
