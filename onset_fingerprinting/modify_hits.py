import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
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
    clicked_line = select(event, vlines)

    if event.button is MouseButton.LEFT:
        if (selected_line is not None) and (clicked_line == selected_line):
            print(f"C = S {clicked_line.line.get_xdata()}")
            # click again - TODO
            moving = True
        elif clicked_line is not None:
            print(f"Selected line, set moving {clicked_line.line.get_xdata()}")
            set_selected(clicked_line)
            moving = True
        else:
            print(f"Setting new on release {event.xdata}")
            new_on_release = True


def on_release(event):
    global selected_line, new_on_release, moving
    moving = False
    if event.button is MouseButton.LEFT:
        if new_on_release and (toolbar.mode is toolbar_mode.NONE):
            print("New line on release")
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
                print("Removing selected line")
                selected_line.line.remove()
                lines.remove(selected_line)
                set_selected(None)
        case " ":
            if selected_line is not None:
                x = int(selected_line.line.get_xdata()[0])
                sd.play(audio[x : x + int(sr / 2)], samplerate=sr)


def on_motion(event):
    global selected_line, new_on_release
    new_on_release = False
    if event.xdata is not None:
        text_var.set(round(event.xdata))
        # If we're panning or zooming we don't want to move a line
        if (
            moving
            and (toolbar.mode is toolbar_mode.NONE)
            and (selected_line is not None)
        ):
            selected_line.line.set_xdata([event.xdata, event.xdata])
            fig.canvas.draw()


# Connect the events to the event handlers
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("key_press_event", on_key)

# create tkinter window
root = tk.Tk()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Show sample index at pointer
text_var = tk.StringVar()
text_entry = tk.Label(root, textvariable=text_var)
text_entry.pack()
opt.setup_tk(root)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


tk.mainloop()
