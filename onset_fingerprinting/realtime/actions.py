import queue
from collections import deque
from dataclasses import KW_ONLY, dataclass, field

from loopmate.actions import Action, Effect, Sample, Trigger

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
class Action:
    """
    Action that triggers if a specific location is hit.
    """

    # Allow for formulations in both spherical and cartesian coordinates
    bounds: tuple
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
