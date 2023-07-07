import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

# Create the figure
fig, ax = plt.subplots()


x = np.linspace(0, 10, 1000)
y = np.sin(x)

# plot waveform
(line,) = ax.plot(x, y, picker=5)


def on_combobox_select(event):
    global selected_line
    selected_line.meta[event.widget.widgetname] = event.widget.get()


@dataclass
class Options:
    meta: dict
    zones: list = field(init=False)
    conditions: list = field(init=False)
    combos: dict = field(init=False)

    def __post_init__(self):
        self.zones = self.meta["zones"]
        self.conditions = self.meta["conditions"]
        self.combos = {}

    def setup_tk(self, frame):
        label = tk.Label(frame, text="Zone")
        label.pack()

        cb = ttk.Combobox(
            frame, name="zone", values=self.zones, state="readonly"
        )
        cb.bind("<<ComboboxSelected>>", on_combobox_select)
        cb.pack()
        self.combos["zone"] = cb

        for condition in self.conditions:
            label = tk.Label(frame, text=condition)
            label.pack()
            cb = ttk.Combobox(
                frame,
                name=str(condition),
                values=self.conditions[condition],
                state="readonly",
            )
            cb.bind("<<ComboboxSelected>>", on_combobox_select)
            cb.pack()
            self.combos[condition] = cb

    def set_meta(self, line):
        if line is None:
            for x in self.combos:
                self.combos[x].set("")
        else:
            for x in line.meta:
                print(line.meta, x, self.combos)
                self.combos[x].set(line.meta[x])


@dataclass
class LineMeta:
    line: Line2D
    # Make sure this meta is the same as the one that'll be setup in Options
    meta: dict

    def export_meta(self):
        meta = self.meta.copy()
        meta["onset_start"] = round(self.line.get_xdata()[0])
        return meta


meta = {"zones": ["a", "b"], "conditions": {"isolated": ["true", "false"]}}
opt = Options(meta)

# indices for vertical lines
vlines_indices = [2, 4, 6, 8]
vlines = [ax.axvline(x, color="green") for x in vlines_indices]
metas = [
    {"zone": "a", "isolated": "true"},
    {"zone": "b", "isolated": "true"},
    {"zone": "b", "isolated": "false"},
    {"zone": "a", "isolated": "false"},
]

lines = [LineMeta(line, meta) for line, meta in zip(vlines, metas)]

# Global variables
selected_line = None
last_meta = metas[0]
new_on_release = False
moving = False


def select(event, vlines):
    for lm in lines:
        if lm.line.contains(event)[0]:
            return lm
    return None


def set_selected(line):
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
        if new_on_release:
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


def on_motion(event):
    global selected_line, new_on_release
    new_on_release = False
    if event.xdata is not None:
        text_var.set(round(event.xdata))
        if moving and selected_line is not None:
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

tk.mainloop()
