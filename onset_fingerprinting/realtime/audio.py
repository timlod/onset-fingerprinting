import queue
from collections import deque

import numpy as np
import sounddevice as sd
from loopmate.utils import CLAVE, StreamTime, channels_to_int
from onset_fingerprinting import multilateration
from onset_fingerprinting.detection import AmplitudeOnsetDetector
from onset_fingerprinting.realtime import config
from onset_fingerprinting.realtime.actions import Actions, Bounds, Location


class PlayRec:
    """
    Main class to set up the looper.  Creates the sd.Stream, holds loop anchor
    and list of audio tracks to loop, and the global action queue.
    """

    def __init__(self, recording, ml_conf, fx, model=None):
        self.current_index = 0
        self.rec = recording
        # Always record audio buffers so we can easily look back for loopables
        self.rec_audio = self.rec.audio

        # Global actions applied to fully mixed audio
        self.actions = Actions()

        self.stream = sd.Stream(
            samplerate=config.SR,
            device=config.DEVICE,
            channels=config.CHANS_IN_OUT,
            callback=self._get_callback(),
            latency=config.LATENCY,
            blocksize=config.BLOCKSIZE,
        )
        self.callback_time = None
        self.last_out = deque(maxlen=20)

        self.od = AmplitudeOnsetDetector(
            config.N_CHANNELS,
            config.BLOCKSIZE,
            hipass_freq=0,
            fast_ar=(0.3, 800),
            slow_ar=(8000, 8000),
            on_threshold=0.45,
            off_threshold=0.45,
            cooldown=1323,
            sr=config.SR,
            backtrack=False,
            backtrack_buffer_size=2 * config.BLOCKSIZE,
            backtrack_smooth_size=1,
        )
        self.m = multilateration.Multilaterate3D(
            sensor_locations=ml_conf["sensor_locations"],
            sr=config.SR,
            medium=ml_conf["medium"],
            c=ml_conf["c"],
            model=model,
        )
        self.fx = fx

    def detect_hits(self, audio):
        c, d, r = self.od(audio)
        if len(c) > 0:
            d = [self.current_index + x for x in d]
            idx = np.argsort(d)
            for i in idx:
                # print(f"Index: {d[i]} on channel {c[i]}")
                res = self.m.locate(c[i], d[i], self.rec_audio)
                if res is not None:
                    res = Location(*res, radius=self.m.radius)
                    print(f"Result: {res}")
                    return res
        return None

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(indata, outdata, frames, time, status):
            """sounddevice callback.  See
            https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.Stream

            Note that frames refers to the number of audio samples (not
            renaming due to sd convention only)
            """
            if status:
                print(status)

            # These times/frame refer to the block that is processed in this
            # callback
            self.callback_time = StreamTime(time, self.current_index)

            # Copy necessary as indata arg is passed by reference
            indata = indata.copy()
            self.rec_audio.write(indata[:, config.CHANNELS])
            # I think this will write an empty sound file in the beginning
            if self.rec_audio.write_counter < frames:
                self.rec.data.analysis_action = 3

            res = self.detect_hits(indata)

            # Store last output buffer to potentially send a slightly delayed
            # version to headphones (to match the speaker sound latency). We do
            # this before running actions such that we can mute the two
            # separately
            # TODO: Define mixing function, remove scale
            outdata[:] = indata[:, :2] * 2
            self.last_out.append((self.callback_time, outdata.copy()))
            if res is not None:
                self.actions.run(outdata, res)

            for fx in self.fx:
                outdata[:] = fx(outdata[:], config.SR, frames, reset=False)

            # Essentially this will be the last index, or the index relative to
            # the current audio buffer inside rec_audio (as its counter will
            # always be updated right after writing)
            self.current_index += frames

        return callback

    def start(self, restart=False):
        """Start stream."""
        self.stream.stop()
        if restart:
            self.current_index = 0
        self.stream.start()

    def stop(self):
        """Stop stream."""
        self.stream.stop()

    def event_counter(self) -> (int, int):
        """Return the recording counter location corresponding to the time when
        this function was called, as well as the offset samples relative to the
        beginning of the current audio block/frame.
        """
        t = self.stream.time
        samples_since = round(self.callback_time.timediff(t) * config.SR)
        return (
            self.rec_audio.counter
            + samples_since
            + round(self.callback_time.input_delay * config.SR)
        ), samples_since
