### Interactively plot multi-channel POSD onsets, to manually verify/fix
# Opens an interactive matplotlib plot
#
# Usage: See python modify_hits.py --help

import argparse
import json
import tkinter as tk
from pathlib import Path

import matplotlib
import numpy as np
import sounddevice as sd
import soundfile as sf

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.backend_bases import MouseButton, _Mode as toolbar_mode
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.lines import Line2D

toolbar = None
current_x_label = None
progress_label = None
root = None


def set_selected(idx):
    global selected_idx
    if selected_idx is not None:
        lines[selected_idx].set_color("darkgrey")
    selected_idx = idx
    if selected_idx is not None:
        lines[selected_idx].set_color("red")
    fig.canvas.draw()


def update_window(center=None):
    """Update visible window and y scaling."""
    global window_center
    if center is not None:
        window_center = center
    start = max(0, int(window_center - window_size // 2))
    end = int(window_center + window_size // 2)
    for i, ax in enumerate(axes):
        ax.set_xlim(start, end)
        seg = audio[start:end, i]
        if len(seg) == 0:
            continue
        rng = seg.max() - seg.min()
        margin = rng * 0.1
        ax.set_ylim(seg.min() - margin, seg.max() + margin)
    fig.canvas.draw()


def update_group(idx, reset_zoom=False):
    global lines, window_size, window_center, start
    onset_list = hits[idx]["onset_start"]
    for i, (line, onset) in enumerate(zip(lines, onset_list)):
        if onset < 0:
            # display at minimum valid onset
            valid = [o for o in onset_list if o >= 0]
            onset_disp = valid[0] if valid else 0
            line.set_linestyle("--")
            line.set_color("orange")
        else:
            onset_disp = onset
            line.set_linestyle("-")
            line.set_color("darkgrey")
        line.set_xdata([onset_disp, onset_disp])

    valid = [o for o in onset_list if o >= 0]
    if valid:
        start = min(valid) - tolerance[0]
        end = max(valid) + tolerance[1]
        window_center = (start + end) / 2
        if reset_zoom or window_size is None:
            window_size = max(end - start, 1)
        update_window()
    if progress_label:
        progress_label.config(text=f"{idx + 1} / {len(hits)}")


def store_current(idx):
    hits[idx]["onset_start"] = [int(l.get_xdata()[0]) for l in lines]


def save_data():
    store_current(current_idx)
    sess["hits"] = hits
    with open(args.data_dir / f"{args.session}-mod.json", "w") as f:
        json.dump(sess, f)


def on_close(event=None):
    sd.stop()
    save_data()
    if root is not None:
        root.quit()
        root.destroy()


def on_click(event):
    global moving
    if event.inaxes is None or event.inaxes not in axes:
        return
    idx = np.where(axes == event.inaxes)[0][0]
    if event.button is MouseButton.LEFT:
        set_selected(idx)
        if event.xdata is not None:
            lines[idx].set_xdata([event.xdata, event.xdata])
            fig.canvas.draw()
        moving = True
    elif event.button is MouseButton.RIGHT:
        set_selected(None)


def on_release(event):
    global moving
    moving = False


def on_motion(event):
    if event.xdata is None:
        return
    current_x_label.config(text=round(event.xdata))
    if (
        moving
        and selected_idx is not None
        and toolbar.mode is toolbar_mode.NONE
    ):
        x = event.xdata
        lines[selected_idx].set_xdata([x, x])
        fig.canvas.draw()


def on_key(event):
    global current_idx, window_size, start
    match event.key:
        case "f":
            store_current(current_idx)
            if current_idx < len(hits) - 1:
                current_idx += 1
                update_group(current_idx)
        case "b":
            store_current(current_idx)
            if current_idx > 0:
                current_idx -= 1
                update_group(current_idx)
        case "+":
            window_size = max(int(window_size / 1.5), 10)
            update_window()
        case "-":
            window_size = int(window_size * 1.5)
            update_window()
        case " ":
            sd.play(audio[start : start + int(sr * 1.5)], samplerate=sr)
        case "q":
            on_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Modify multichannel onset annotations.

In the plot, click in the plot to move the onsets and press:
    f   : move to next onset group
    b   : move to the previous onset group
    +   : to zoom in
    -   : to zoom out
    SPC : to play the audio starting at the current window        
    q   : to save the session JSON (as $SESSION-mod.json)
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("data_dir", type=Path, help="Data directory")
    parser.add_argument("session", type=str, help="Session name")
    parser.add_argument(
        "--tolerance",
        "-t",
        type=int,
        nargs=2,
        default=(64, 128),
        metavar=("LEFT", "RIGHT"),
        help="tolerances to add left and right of onset group",
    )
    args = parser.parse_args()

    global hits, sess
    with open(args.data_dir / f"{args.session}.json") as f:
        sess = json.load(f)
    hits = sess["hits"]

    global audio, sr, fig, axes, lines, current_idx, selected_idx, moving, window_size, window_center, tolerance
    tolerance = args.tolerance
    audio, sr = sf.read(args.data_dir / f"{args.session}.wav")
    n = audio.shape[0]
    subsampling = 2

    fig, axes = plt.subplots(audio.shape[1], 1, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, ax in enumerate(axes):
        ax.plot(
            range(0, n, subsampling),
            audio[:n:subsampling, i],
            picker=5,
            c=cm["tab10"](i),
        )

    global lines, window_size, current_idx, selected_idx, moving, window_center
    lines = []
    for ax in axes:
        line = ax.axvline(0, color="darkgrey")
        lines.append(line)

    window_size = None
    start = 0
    window_center = 0
    current_idx = 0
    selected_idx = None
    moving = False

    def start():
        global toolbar, current_x_label, progress_label, root
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("button_release_event", on_release)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("key_press_event", on_key)

        root = tk.Tk()
        root.protocol("WM_DELETE_WINDOW", on_close)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        current_x_label = tk.Label(root, text="")
        current_x_label.pack()
        progress_label = tk.Label(root, text="")
        progress_label.pack()

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        fig.canvas.toolbar.push_current()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        update_group(0, reset_zoom=True)
        root.mainloop()

    current_x_label = None
    progress_label = None

    start()
