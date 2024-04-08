import threading
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pedalboard
import rtmidi
import sounddevice as sd
import soundfile as sf
from loopmate import recording
from loopmate.actions import BackCaptureTrigger, Effect, RecordTrigger
from onset_fingerprinting.realtime import audio, config

## TODOs:
# record all required channels somehow
# playback only the stuff we want to change, e.g. samples, effects
# setup rt onset detection stuff
# implement some things to control/map


def plan_callback(pr: audio.PlayRec):
    """Callback which picks up triggers/actions from the plan queue.

    :param pr: PlayRec object containing Actions
    """
    while True:
        print("plan")
        trigger = pr.actions.plans.get()
        if isinstance(trigger, RecordTrigger):
            print("Record in plan_callback")
            # TODO: this will run into trouble if result_type == 8
            if pr.rec.data.result_type == 0:
                pr.start_recording()
            else:
                pr.stop_recording()
            continue
        elif isinstance(trigger, BackCaptureTrigger):
            pr.backcapture(trigger.n_loops)
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
    ml_conf = config.load_setup(
        Path(__name__).parent.parent.parent / "data" / "demo" / "conf.json"
    )
    with recording.RecAudio(config.REC_N, config.N_CHANNELS) as rec:
        ap = Process(target=analysis_target)
        ap2 = Process(target=ondemand_target)
        ap.start()
        ap2.start()

        print(sd.query_devices())
        pr = audio.PlayRec(rec, ml_conf)
        pr.start()

        # Some example effects that can be applied. TODO: make more
        # interesting/intuitive
        ps = pedalboard.PitchShift(semitones=-6)
        ds = pedalboard.Distortion(drive_db=20)
        delay = pedalboard.Delay(0.8, 0.1, 0.3)
        limiter = pedalboard.Limiter()
        pr.actions.append(Effect(0, 10000000, lambda x: limiter(x, config.SR)))

        plan_thread = threading.Thread(target=plan_callback, args=(pr,))
        plan_thread.start()
        ap.join()
        ap2.join()
        plan_thread.join()
        sd.sleep(10)
