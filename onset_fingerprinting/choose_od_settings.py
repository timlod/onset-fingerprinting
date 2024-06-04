from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.widgets import Slider

from onset_fingerprinting import detection

# data_dir = Path("../data/calibration/2023-03-19")
# data = []
# for sensor in ["OH L", "OH R", "Snare Top"]:
#     x, sr = sf.read(
#         data_dir / "snare" / "calib" / ("snare_2cm" + sensor + ".wav"),
#         dtype=np.float32,
#     )
#     data.append(x)
# data = np.array(data).T
# audio_calib = np.ascontiguousarray(data).copy()[: sr * 20]
# close_channel = 2

data_dir = Path("../data/location/Recordings3")
data, sr = sf.read(data_dir / "Setup 2" / "155 hits.wav", dtype=np.float32)
audio_calib = data[15 * sr :, :3][: sr * 10]
close_channel = None


class InteractivePlot:
    def __init__(self, audio_calib: np.ndarray, sr: int):
        self.audio_calib = audio_calib
        self.sr = sr
        self.hipass_freq = 1000
        self.fast_ar = (1, 900)
        self.slow_ar = (8000, 8000)
        self.on_thresholds = [0.45, 0.45, 0.45]
        self.floor = -70
        self.off_threshold = 0.2
        self.cooldown = 0

        self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35)
        self.plot_initial_data()

        self.ax_hipass = plt.axes([0.1, 0.25, 0.8, 0.03])
        self.ax_fast_ar_0 = plt.axes([0.1, 0.2, 0.35, 0.03])
        self.ax_fast_ar = plt.axes([0.55, 0.2, 0.35, 0.03])
        self.ax_slow_ar_0 = plt.axes([0.1, 0.15, 0.35, 0.03])
        self.ax_slow_ar = plt.axes([0.55, 0.15, 0.35, 0.03])
        self.ax_on_threshold_1 = plt.axes([0.1, 0.1, 0.55, 0.03])
        self.ax_on_threshold_2 = plt.axes([0.1, 0.05, 0.55, 0.03])
        self.ax_on_threshold_3 = plt.axes([0.1, 0, 0.55, 0.03])
        self.ax_floor = plt.axes([0.78, 0.1, 0.15, 0.03])
        self.ax_off_threshold = plt.axes([0.78, 0.05, 0.15, 0.03])
        self.ax_cooldown = plt.axes([0.78, 0.0, 0.15, 0.03])

        self.s_fast_ar_0 = Slider(
            self.ax_fast_ar_0, "Fast AR 0", 0, 20, valinit=self.fast_ar[0]
        )
        self.s_slow_ar_0 = Slider(
            self.ax_slow_ar_0,
            "Slow AR 0",
            1000,
            20000,
            valinit=self.slow_ar[0],
        )

        self.s_hipass = Slider(
            self.ax_hipass, "Hipass Freq", 0, 5000, valinit=self.hipass_freq
        )
        self.s_fast_ar = Slider(
            self.ax_fast_ar, "Fast AR", 1, 5000, valinit=self.fast_ar[1]
        )
        self.s_slow_ar = Slider(
            self.ax_slow_ar, "Slow AR", 1000, 20000, valinit=self.slow_ar[1]
        )
        self.s_on_threshold_1 = Slider(
            self.ax_on_threshold_1,
            "On Threshold 1",
            0.0,
            1.0,
            valinit=self.on_thresholds[0],
        )
        self.s_on_threshold_2 = Slider(
            self.ax_on_threshold_2,
            "On Threshold 2",
            0.0,
            1.0,
            valinit=self.on_thresholds[1],
        )
        self.s_on_threshold_3 = Slider(
            self.ax_on_threshold_3,
            "On Threshold 3",
            0.0,
            1.0,
            valinit=self.on_thresholds[2],
        )
        self.s_floor = Slider(
            self.ax_floor,
            "Floor",
            -100,
            -20,
            valinit=self.floor,
        )
        self.s_off_threshold = Slider(
            self.ax_off_threshold,
            "Off Threshold",
            0.0,
            1.0,
            valinit=self.off_threshold,
        )
        self.s_cooldown = Slider(
            self.ax_cooldown,
            "Cooldown",
            0,
            int(0.1 * sr),
            valinit=self.cooldown,
        )

        self.s_hipass.on_changed(self.update)
        self.s_fast_ar_0.on_changed(self.update)
        self.s_slow_ar_0.on_changed(self.update)
        self.s_fast_ar.on_changed(self.update)
        self.s_slow_ar.on_changed(self.update)
        self.s_on_threshold_1.on_changed(self.update)
        self.s_on_threshold_2.on_changed(self.update)
        self.s_on_threshold_3.on_changed(self.update)
        self.s_floor.on_changed(self.update)
        self.s_off_threshold.on_changed(self.update)
        self.s_cooldown.on_changed(self.update)

    def plot_initial_data(self):
        cm = plt.colormaps["tab10"].colors
        x = self.audio_calib
        self.lines = [
            ax.plot(x[::100, i], c=cm[i])[0]
            for i, ax in enumerate(self.axs[:3])
        ]
        self.vlines = [
            ax.vlines([], x.min(), x.max(), color="red") for ax in self.axs[:3]
        ]
        self.group_vlines = self.axs[3].vlines(
            [], x.min(), x.max(), color="red"
        )
        for i, ax in enumerate(self.axs[:3]):
            ax.set_title(f"Channel {i+1}")
            ax.set_xlabel("Samples (subsampled)")
            ax.set_ylabel("Amplitude")
        self.axs[3].set_title("Grouped Onsets")
        self.axs[3].set_xlabel("Samples (subsampled)")
        self.axs[3].set_ylabel("Amplitude")
        self.axs[3].plot(
            x[::100, 0]
        )  # Plotting the first channel for grouped onsets

    def plot_vlines(self, onsets, channels):
        for vline in self.vlines:
            if vline:
                vline.remove()
        for i, ax in enumerate(self.axs[:3]):
            channel_onsets = onsets[channels == i]
            self.vlines[i] = ax.vlines(
                channel_onsets / 100,
                self.audio_calib.min(),
                self.audio_calib.max(),
                color="red",
            )
        plt.draw()

    def plot_grouped_vlines(self, grouped_onsets):
        if isinstance(self.group_vlines, list):
            for line in self.group_vlines:
                line.remove()
        else:
            self.group_vlines.remove()
        self.group_vlines = self.axs[3].vlines(
            grouped_onsets[:, 0] / 100,
            self.audio_calib.min(),
            self.audio_calib.max(),
            color="red",
        )
        plt.draw()

    def update(self, val):
        self.hipass_freq = self.s_hipass.val
        self.fast_ar = (self.s_fast_ar_0.val, self.s_fast_ar.val)
        self.slow_ar = (self.s_slow_ar_0.val, self.s_slow_ar.val)
        self.on_thresholds = [
            self.s_on_threshold_1.val,
            self.s_on_threshold_2.val,
            self.s_on_threshold_3.val,
        ]
        self.floor = self.s_floor.val
        self.off_threshold = self.s_off_threshold.val
        self.cooldown = self.s_cooldown.val

        cf, of = detection.detect_onsets_amplitude(
            self.audio_calib,
            block_size=128,
            floor=self.floor,
            hipass_freq=self.hipass_freq,
            fast_ar=self.fast_ar,
            slow_ar=self.slow_ar,
            on_threshold=np.array(self.on_thresholds),
            off_threshold=self.off_threshold,
            cooldown=self.cooldown,
            sr=self.sr,
            backtrack=True,
            backtrack_buffer_size=256,
            backtrack_smooth_size=1,
        )

        oc = detection.find_onset_groups(
            of, cf, 600, close_channel=close_channel
        )
        self.plot_vlines(np.array(of), np.array(cf))
        self.plot_grouped_vlines(oc)


# Create an instance of the interactive plot
interactive_plot = InteractivePlot(audio_calib, sr)
plt.show()
