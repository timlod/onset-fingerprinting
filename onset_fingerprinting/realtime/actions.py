import queue
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
from typing import Optional

import numpy as np
from loopmate.actions import Action, Effect, Sample, Trigger
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
    # If True, loop this action instead of consuming it
    priority: int = 3
    # Consuming this action will 'spawn'/queue this new action
    spawn: Action | None = None

    def __post_init__(self):
        if self.end > self.start:
            self.n = self.end - self.start
        else:
            self.n = self.start + self.loop_length - self.end

        # Current sample !inside action between start and end
        # don't mix up with current_index in loop!
        self.current_sample = 0
        self.consumed = False

    def trigger(self, location):
        """Run at every step to signal when the out buffer enters the start/end
        boundaries of this action.

        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        pass

    def run(self, data, location):
        """Run action on incoming audio and updates internal counters
        accordingly.

        :param data: audio buffer
        """
        pass

    def __lt__(self, other):
        return self.priority < other.priority

    def do(self, outdata, current_index):
        """Perform manipulations on the output buffer.

        :param outdata: output buffer
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

    def append(self, action: Action | Trigger):
        self.actions.append(action)

    def prepend(self, action: Action | Trigger):
        self.actions.insert(0, action)

    def run(self, outdata, location):
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
            if isinstance(action, Trigger):
                print(f"Trigger {action}, {location}")
                action.run(self)
                if action.consumed:
                    print(self.plans)
                    if not action.loop:
                        self.actions.remove(action)
                    if action.spawn is not None:
                        self.actions.append(action.spawn)
                continue

            # Actions
            action.run(outdata)
            if action.consumed:
                print(f"consumed {action}")
                self.actions.remove(action)
                if action.spawn is not None:
                    print(f"Spawning {action.spawn}")
                    self.actions.append(action.spawn)
