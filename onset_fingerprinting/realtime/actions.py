from __future__ import annotations

import queue
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
from typing import Callable, Optional

import numpy as np
from loopmate.actions import CrossFade, Sample
from onset_fingerprinting.multilateration import cartesian_to_polar

# Reverb/delay combo where angle works with the two
# feedback from bottom to sides
# top zone to activate some effects, perhaps filter

# Allow for blending of actions by defining e.g. a radius around the given
# bounds


# Use actions but instead of loop positions the spherical position is used to
# change an effect
class DelayVerb:
    pass


@dataclass
class Location:
    x: float
    y: float

    def __post_init__(self):
        self.r, self.phi = cartesian_to_polar(self.x, self.y)

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
        print(cart_check, polar_check)
        return cart_check and polar_check


@dataclass
class Action:
    """
    Action that triggers if a specific location is hit.
    """

    bounds: list[Bounds]
    _: KW_ONLY
    countdown: int = 0
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

    def run(self, outdata, location: Location):
        """Run all actions (to be called once every callback)

        :param outdata: outdata as passed into sd callback (will fill portaudio
            buffer)
        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        # Activate actions (puts them in active queue)
        for action in self.actions:
            if action.trigger(location):
                self.active.put_nowait(action)

        while not self.active.empty():
            action = self.active.get_nowait()
            action.run(outdata)
            if action.consumed:
                print(f"consumed {action}")
                if not action.loop:
                    self.actions.remove(action)
                if action.spawn is not None:
                    print(f"Spawning {action.spawn}")
                    self.actions.append(action.spawn)
