import numpy as np
import pedalboard
import rtmidi
import sounddevice as sd
import soundfile as sf

import threading
from multiprocessing import Process

from loopmate import config, recording
from loopmate.actions import (
    BackCaptureTrigger,
    Effect,
    RecordTrigger,
)
from loopmate.loop import Audio, ExtraOutput, Loop

## TODOs:
# record all required channels somehow
# playback only the stuff we want to change, e.g. samples, effects
# setup rt onset detection stuff
# implement some things to control/map


def plan_callback(loop: Loop):
    """Callback which picks up triggers/actions from the plan queue.

    :param loop: Loop object containing Actions
    """
    while True:
        print("plan")
        trigger = loop.actions.plans.get()
        if isinstance(trigger, RecordTrigger):
            print("Record in plan_callback")
            # TODO: this will run into trouble if result_type == 8
            if loop.rec.data.result_type == 0:
                loop.start_recording()
            else:
                loop.stop_recording()
            continue
        elif isinstance(trigger, BackCaptureTrigger):
            loop.backcapture(trigger.n_loops)
            continue
        elif isinstance(trigger, bool):
            break


def analysis_target():
    """
    target function for the multiprocessing.Process which will run ongoing
    analysis on the audio which is constantly recorded.
    """
    with recording.RecAnalysis(config.REC_N, config.N_CHANNELS) as rec:
        rec.run()
    print("done analysis")


def ondemand_target():
    """target function for the multiprocessing.Process which will run
    analysis like onset quantization or BPM estimation on demand.
    """
    with recording.AnalysisOnDemand(config.REC_N, config.N_CHANNELS) as rec:
        rec.run()
    print("done ondemand")


if __name__ == "__main__":
    with recording.RecAudio(config.REC_N, config.N_CHANNELS) as rec:
        ap = Process(target=analysis_target)
        ap2 = Process(target=ondemand_target)
        ap.start()
        ap2.start()

        print(sd.query_devices())
        clave, _ = sf.read("../data/clave.wav", dtype=np.float32)
        clave = np.concatenate(
            (
                1 * clave[:, None],
                np.zeros((config.SR - len(clave), 1), dtype=np.float32),
            )
        )
        loop = Loop(rec, Audio(clave))
        loop.start()

        if config.HEADPHONE_DEVICE != config.DEVICE:
            hl = ExtraOutput(loop)

        # Some example effects that can be applied. TODO: make more
        # interesting/intuitive
        ps = pedalboard.PitchShift(semitones=-6)
        ds = pedalboard.Distortion(drive_db=20)
        delay = pedalboard.Delay(0.8, 0.1, 0.3)
        limiter = pedalboard.Limiter()
        loop.actions.append(
            Effect(0, 10000000, lambda x: limiter(x, config.SR))
        )

        plan_thread = threading.Thread(target=plan_callback, args=(loop,))
        plan_thread.start()
        ap.join()
        ap2.join()
        plan_thread.join()
        sd.sleep(10)
