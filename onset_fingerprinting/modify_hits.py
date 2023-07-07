import argparse
import json
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from matplotlib.backend_bases import MouseButton, _Mode as toolbar_mode
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.lines import Line2D


@dataclass
class LineMeta:
    line: Line2D
    # Make sure this meta is the same as the one that'll be setup in Options
    meta: dict

    def export_meta(self):
        meta = self.meta.copy()
        meta["onset_start"] = round(self.line.get_xdata()[0])
        return meta


@dataclass
class MetaBoxes:
    meta: dict
    combos: dict = field(init=False)

    def __post_init__(self):
        self.combos = {}

    def setup_tk(self, frame):
        """Setup the combobox objects and pack them onto the frame.

        :param frame: top-level TK widget
        """
        label = tk.Label(frame, text="Zone")
        label.pack()
        cb = ttk.Combobox(
            frame,
            name="zone",
            values=[x for x in self.meta["zones"]],
            state="readonly",
        )
        cb.bind("<<ComboboxSelected>>", on_combobox_select)
        cb.pack()
        self.combos["zone"] = cb

        for condition in self.meta["conditions"]:
            label = tk.Label(frame, text=condition)
            label.pack()
            cb = ttk.Combobox(
                frame,
                name=str(condition),
                values=self.meta["conditions"][condition],
                state="readonly",
            )
            cb.bind("<<ComboboxSelected>>", on_combobox_select)
            cb.pack()
            self.combos[condition] = cb

        onset_start = tk.StringVar()
        os = tk.Label(root, textvariable=onset_start)
        os.pack()
        self.combos["onset_start"] = onset_start

    def set_meta(self, line: LineMeta | None):
        """Set the combobox contents accordign to a line's metadata.

        :param line: LineMeta object
        """
        if line is None:
            for x in self.combos:
                self.combos[x].set("")
        else:
            for x in line.meta:
                if x in self.combos:
                    self.combos[x].set(line.meta[x])


def on_combobox_select(event):
    """Sets the metadata of an onset if it was changed in the widget.

    :param event: combobox widget change event
    """
    global selected_line
    # This will only be called if selected_line is not None, so this is fine
    selected_line.meta[event.widget.widgetname] = event.widget.get()


def select_close_line(event) -> LineMeta | None:
    """Select the first close line to event according to the picker radius of
    the figure. Returns None if none found.

    :param event: click event
    """
    for lm in lines:
        if lm.line.contains(event)[0]:
            return lm
    return None


def set_selected(line: LineMeta | None):
    """Visually select a line and redraw.

    :param line: line to mark or None to stop a visual selection
    """
    global selected_line, last_meta
    # Unmark previous selection
    if selected_line is not None:
        selected_line.line.set_color("green")
    # Set new selection and mark if it's not None
    selected_line = line
    if selected_line is not None:
        last_meta = selected_line.meta
        selected_line.line.set_color("red")
    opt.set_meta(selected_line)
    fig.canvas.draw()


def on_click(event):
    global selected_line, moving, new_on_release
    clicked_line = select_close_line(event)

    if event.button is MouseButton.LEFT:
        if (selected_line is not None) and (clicked_line == selected_line):
            moving = True
        elif (clicked_line is not None) and (
            toolbar.mode is toolbar_mode.NONE
        ):
            set_selected(clicked_line)
            moving = True
        else:
            new_on_release = True


def on_release(event):
    global selected_line, new_on_release, moving
    moving = False
    if event.button is MouseButton.LEFT:
        if new_on_release and (toolbar.mode is toolbar_mode.NONE):
            new_line = LineMeta(
                ax.axvline(event.xdata, color="red"), last_meta
            )
            lines.append(new_line)
            set_selected(new_line)
    elif event.button is MouseButton.RIGHT:
        set_selected(None)
    new_on_release = False


def on_key(event):
    global selected_line
    match event.key:
        case "d":
            if selected_line is not None:
                selected_line.line.remove()
                lines.remove(selected_line)
                set_selected(None)
        case " ":
            if selected_line is not None:
                x = int(selected_line.line.get_xdata()[0])
                sd.play(audio[x : x + int(sr / 2)], samplerate=sr)
        case "z":
            if selected_line is not None:
                onset = selected_line.meta["onset_start"]
                ax.set_xlim((onset - sr // 4, onset + sr * 2))
                fig.canvas.toolbar.push_current()
                fig.canvas.draw()
        case "f":
            xlims = ax.get_xlim()
            ax.set_xlim((xlims[0] + sr // 8, xlims[1] + sr // 8))
            fig.canvas.toolbar.push_current()
            fig.canvas.draw()
        case "q":
            out = []
            for line in lines:
                out.append(line.export_meta())
            out = sorted(out, key=lambda x: x["onset_start"])
            with open(args.data_dir / f"{args.session}-mod.json", "w") as f:
                json.dump(dict_long_to_wide(out), f)


def on_motion(event):
    global selected_line, new_on_release
    new_on_release = False
    if event.xdata is not None:
        current_pointer.set(round(event.xdata))
        # If we're panning or zooming we don't want to move a line
        if (
            moving
            and (toolbar.mode is toolbar_mode.NONE)
            and (selected_line is not None)
        ):
            selected_line.line.set_xdata([event.xdata, event.xdata])
            fig.canvas.draw()


def dict_long_to_wide(input_list: list) -> dict:
    """Transform a list of dictionaries to a dictionary of lists.

    :param input_dict: input dictionary
    """
    output: dict[str, list] = {}
    for item in input_list:
        for key, value in item.items():
            if key not in output:
                output[key] = []
            output[key].append(value)
    return output


def dict_wide_to_long(input_dict: dict) -> list:
    """Transform a 'wide' dictionary of lists to a list of dictionaries with
    keys duplicated for each observation.

    :param input_dict: input dictionary in list format
    """

    # Get the length of the lists by taking the length of the values of the first key
    list_len = len(next(iter(input_dict.values())))

    # Initialize an empty list to hold the output dictionaries
    output_list = []

    # Iterate over the range of the list length
    for i in range(list_len):
        # Create a new dictionary for each index,
        # mapping each key in the input_dict to the corresponding value at the current index
        output_dict = {key: input_dict[key][i] for key in input_dict.keys()}
        # Add the new dictionary to the output list
        output_list.append(output_dict)

    return output_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify detected onset metadata."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="The data directory containing both .wav and .json files",
    )
    parser.add_argument(
        "session",
        type=str,
        help="Name of the session (<session>.json needs to exist in data_dir)",
    )
    parser.add_argument(
        "--instrument",
        "-i",
        default="snare",
        required=False,
        type=str,
        help="Name of instrument used in instrument.json (default snare)",
    )
    parser.add_argument(
        "--channel",
        "-c",
        default="OP",
        required=False,
        type=str,
        help="Name of channel to load for visualization. (default OP)",
    )

    args = parser.parse_args()

    with open(args.data_dir / "instruments.json") as f:
        inst = json.load(f)[args.instrument]
    opt = MetaBoxes(inst)

    with open(args.data_dir / f"{args.session}.json") as f:
        sess = json.load(f)
        hits = sess["hits"]

    audio, sr = sf.read(args.data_dir / f"{args.session}_{args.channel}.wav")
    n = len(audio)
    # Use this when plotting longer files and experiencing slowdown
    subsampling = 2

    fig, ax = plt.subplots()
    (line,) = ax.plot(
        range(0, n, subsampling), audio[:n:subsampling], picker=5
    )

    vlines = [ax.axvline(x, color="green") for x in hits["onset_start"]]
    hits = dict_wide_to_long(hits)
    lines = [LineMeta(line, meta) for line, meta in zip(vlines, hits)]

    # Global variables
    selected_line: LineMeta | None = None
    # last_meta = metas[0]
    last_meta = hits[0]
    new_on_release = False
    moving = False

    # Connect events to event handlers
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Create tkinter window
    root = tk.Tk()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Show sample index at pointer
    current_pointer = tk.StringVar()
    text_entry = tk.Label(root, textvariable=current_pointer)
    text_entry.pack()
    opt.setup_tk(root)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    fig.canvas.toolbar.push_current()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    tk.mainloop()
