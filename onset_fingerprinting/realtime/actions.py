from __future__ import annotations

import queue
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
from typing import Callable, Optional

import numpy as np
import pedalboard
from onset_fingerprinting.multilateration import (
    cartesian_to_polar,
    polar_to_cartesian,
)

# Reverb/delay combo where angle works with the two
# feedback from bottom to sides
# top zone to activate some effects, perhaps filter

# Allow for blending of actions by defining e.g. a radius around the given
# bounds

# TODO: Connect plugins to audio playback - how should this be done?
# - just have them sit permanently on playback for now


def value_in_parameter_range(
    ranges: dict[tuple[float, float], float], value: float
) -> float | None:
    """
    Retrieve the dictionary value for which the input value lies within the
    key's range.

    :param ranges: Dictionary with tuple keys representing ranges and float
        values
    :param value: The float value to test against the range keys in the
        dictionary
    """
    if value == 1:
        return list(ranges.values())[-1]
    for range_key, associated_value in ranges.items():
        if range_key[0] <= value <= range_key[1]:
            return associated_value
    return None


def map_fx_param_range(fx, name, val):
    p = fx.parameters[name]
    return value_in_parameter_range(p.ranges, val)


class ParameterMapper:
    """
    Maps floating-point numbers from an original range to a target range with
    an optional non-linear transformation function.

    :param coordinate: which coordinate to use for this mapping, must be one of
        {x, y, r, phi}
    :param target_names: names of the target parameters to map to
    :param original_range: A tuple (min, max) defining the original range.
    :param target_ranges: A tuple (min, max) defining the target range.
    :param transformation: An optional function that applies a non-linear
        transformation to the scaled value.  Use powers of x, or nth root of x.
    """

    def __init__(
        self,
        coordinate: str,
        target_names: list[str],
        original_range: tuple[float, float],
        target_ranges: list[tuple[float, float]],
        transformation: Optional[Callable[[float], float]] = None,
    ):
        self.coordinate = coordinate
        self.target_names = target_names
        self.original_min, self.original_max = original_range
        self.target_ranges = target_ranges
        self.transformation = transformation

    def __call__(self, x: float) -> float:
        """
        Maps a value from the original range to the target ranges using an
        optional transformation function.

        :param value: The value to map from the original range.
        """
        x_norm = (x - self.original_min) / (
            self.original_max - self.original_min
        )

        if self.transformation:
            x_norm = self.transformation(x_norm)

        return [
            (x_norm * (target_max - target_min)) + target_min
            for target_min, target_max in self.target_ranges
        ]

    @classmethod
    def from_bounds_fx(
        cls,
        bounds: Bounds,
        effect: pedalboard.Plugin,
        coordinate: str,
        parameters: list[str],
        transformation: Optional[Callable[[float], float]] = None,
    ):
        """Create a ParameterMapper to map directly from a given boundary to an
        effect.

        :param bounds: Bounds object that the parameter change should trigger
            on
        :param effect: pedalboard Plugin
        :param coordinate: one of {x, y, r, phi}
        :param parameters: name of the parameters in effect to map to
        """
        assert all(
            [name in effect.parameters for name in parameters]
        ), "FX parameters and given parameter names don't align!"

        original_range = (
            getattr(bounds, f"{coordinate}_min"),
            getattr(bounds, f"{coordinate}_max"),
        )
        # target_ranges = [
        #     effect.parameters[param].range[:2] for param in parameters
        # ]
        target_ranges = [(0, 1) for param in parameters]
        return cls(coordinate, parameters, original_range, target_ranges)

    def from_bounds(
        cls,
        bounds: Bounds,
        coordinate: str,
        target_names: list[str],
        target_ranges: list[tuple[float, float]],
        transformation: Optional[Callable[[float], float]] = None,
    ):
        """Create a ParameterMapper to map directly from a given boundary to an
        effect.

        :param bounds: Bounds object that the parameter change should trigger
            on
        :param effect: pedalboard Plugin
        :param coordinate: one of {x, y, r, phi}
        :param parameters: name of the parameters in effect to map to
        """
        original_range = (
            getattr(bounds, f"{coordinate}_min"),
            getattr(bounds, f"{coordinate}_max"),
        )
        return cls(coordinate, target_names, original_range, target_ranges)


# Use actions but instead of loop positions the spherical position is used to
# change an effect
class DelayVerb:
    pass


@dataclass
class Location:
    x: float = None
    y: float = None
    r: float = None
    phi: float = None
    radius: float = None

    def __post_init__(self):
        if self.x is None:
            self.x, self.y = polar_to_cartesian(self.r, self.phi)
        else:
            self.r, self.phi = cartesian_to_polar(
                self.x, self.y, r=self.radius
            )

    def __repr__(self):
        return f"Location({self.x=}, {self.y=}, {self.r=}, {self.phi=})"


@dataclass
class Bounds:
    """Bounds used to determine where an action will be valid."""

    def __init__(
        self,
        x: Optional[tuple[float, float]] = None,
        y: Optional[tuple[float, float]] = None,
        r: Optional[tuple[float, float]] = None,
        phi: Optional[tuple[float, float]] = None,
    ):
        """Initialize bounds with any combination of x,y,r,phi tuples.  The
        order of the tuples matters only for phi due to its circular nature
        (i.e. for phi, a tuple of [270, 90] is reasonable).

        :param x: bounds for x
        :param y: bounds for y
        :param r: bounds for r (need to be larger than 0)
        :param phi: bounds for phi (min can be larger than max)
        """
        x = sorted(x) if x is not None else (-np.inf, np.inf)
        y = sorted(y) if y is not None else (-np.inf, np.inf)
        r = sorted(r) if r is not None else (-np.inf, np.inf)
        phi = phi if phi is not None else (-np.inf, np.inf)

        self.x_min, self.x_max = x
        self.y_min, self.y_max = y
        self.r_min, self.r_max = r
        self.phi_min, self.phi_max = phi
        self.or_check = True if self.phi_min > self.phi_max else False

    def __contains__(self, location: Location):
        cart_check = (
            self.x_min <= location.x <= self.x_max
            and self.y_min <= location.y <= self.y_max
        )
        if self.or_check:
            polar_check = self.r_min <= location.r <= self.r_max and (
                location.phi >= self.phi_min or location.phi <= self.phi_max
            )
        else:
            polar_check = (
                self.r_min <= location.r <= self.r_max
                and self.phi_min <= location.phi <= self.phi_max
            )
        return cart_check and polar_check


@dataclass
class Action:
    """
    Action that triggers if a specific location is hit.
    """

    bounds: list[Bounds]
    _: KW_ONLY
    countdown: int = 0
    loop: bool = True
    # TODO: make this clearer - right now only relevant for sample playback
    n: int = 0
    # If True, loop this action instead of consuming it
    priority: int = 3
    # Consuming this action will 'spawn'/queue this new action
    spawn: Action | None = None

    def __post_init__(self):
        # Current sample !inside action between start and end
        # don't mix up with current_index in loop!
        self.current_sample = 0
        self.consumed = False

    def trigger(self, location: Location):
        """Run at every hit to signal if the location of the hit corresponds to
        this action.

        :param location: cartesian coordinates of the hit
        """
        for bounds in self.bounds:
            if location in bounds:
                return True
        return False

    def run(self, data: np.ndarray, location: Location):
        """Run action on incoming audio and updates internal counters
        accordingly.

        :param data: audio buffer
        """
        self.do(data, location)
        self.current_sample += len(data)

        if self.current_sample >= self.n:
            if self.loop:
                self.current_sample = 0
            elif self.countdown > 0:
                self.current_sample = 0
                self.countdown -= 1
            else:
                self.consumed = True

    def __lt__(self, other):
        return self.priority < other.priority

    def do(self, data: np.ndarray, location: Location):
        """Perform manipulations on the output buffer.

        :param location: Location of the hit
        """
        raise NotImplementedError("Subclasses need to override this!")

    def cancel(self):
        """Immediately cancel/stop this action."""
        self.current_sample = self.n
        self.loop = False
        self.countdown = 0
        self.consumed = True

    def set_priority(self, priority):
        self.priority = priority


class ParameterChange(Action):
    def __init__(
        self,
        bounds: list[Bounds],
        effect,
        parameter_mappers: list[ParameterMapper],
    ):
        """Initialize action to change fx parameters on triggering."""
        # TODO: currently using loop to indicate non-consumption
        super().__init__(bounds, loop=True)
        self.effect = effect
        self.pms = parameter_mappers
        for pm in self.pms:
            assert all(
                [name in self.effect.parameters for name in pm.target_names]
            ), "FX parameters and ParameterMapper names don't align!"

    def do(self, data, location: Location):
        """Called from within run inside callback. Applies the effect.

        :param data: outdata in callback, modified in-place
        """
        for pm in self.pms:
            mapped_values = pm(getattr(location, pm.coordinate))
            for param, value in zip(pm.target_names, mapped_values):
                print(f"Setting {param} to {value}.")
                # setattr(self.effect, param, value)
                setattr(self.effect.parameters[param], "raw_value", value)

    def cancel(self):
        """Stops effect over the next buffer(s).  Fades out to avoid audio
        popping, and may thus take several callbacks to complete.
        """
        self.current_sample = self.n
        self.loop = False


@dataclass
class Actions:
    # keeps and maintains a queue of actions that are fired in the callback
    max: int = 20
    actions: deque = field(default_factory=deque)
    active: queue.PriorityQueue = field(default_factory=queue.PriorityQueue)
    plans: queue.PriorityQueue = field(default_factory=queue.PriorityQueue)

    def append(self, action: Action):
        self.actions.append(action)

    def prepend(self, action: Action):
        self.actions.insert(0, action)

    def trigger(self, location):
        # Activate actions (puts them in active queue)
        for action in self.actions:
            if action.trigger(location):
                self.active.put_nowait(action)

    def run(self, outdata, location: Location):
        """Run all actions (to be called once every callback)

        :param outdata: outdata as passed into sd callback (will fill portaudio
            buffer)
        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        # TODO: think about how to trigger samples correctly!
        readd = []
        while not self.active.empty():
            action = self.active.get_nowait()
            action.run(outdata, location)
            if action.consumed:
                print(f"consumed {action}")
                ## TODO: think about how to trigger samples properly!
                action.reset()
                # if not action.loop:
                #     self.actions.remove(action)
                if action.spawn is not None:
                    print(f"Spawning {action.spawn}")
                    self.actions.append(action.spawn)
            else:
                readd.append(action)

        for action in readd:
            self.active.put_nowait(action)
