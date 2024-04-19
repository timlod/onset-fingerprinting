import threading
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pedalboard
import rtmidi
import sounddevice as sd
import soundfile as sf
from loopmate.actions import BackCaptureTrigger, Effect, RecordTrigger
from onset_fingerprinting.realtime import actions, audio, config, recording

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
    with recording.RecAnalysis(
        config.REC_N, config.N_CHANNELS, name="rt"
    ) as rec:
        rec.run()
    print("done analysis")


def ondemand_target():
    """target function for the multiprocessing.Process which will run
    analysis like onset quantization or BPM estimation on demand.
    """
    with recording.AnalysisOnDemand(
        config.REC_N, config.N_CHANNELS, name="rt"
    ) as rec:
        rec.run()
    print("done ondemand")


if __name__ == "__main__":
    ml_conf = config.load_setup(
        Path(__file__).parent.parent.parent / "data" / "demo" / "conf.json"
    )
    print(ml_conf)
    with recording.RecAudio(config.REC_N, config.N_CHANNELS, name="rt") as rec:
        # ap = Process(target=analysis_target)
        ap2 = Process(target=ondemand_target)
        # ap.start()
        ap2.start()

        # Some example effects that can be applied. TODO: make more
        # interesting/intuitive
        cc = pedalboard.load_plugin("/usr/lib/vst3/ChowCentaur.vst3")
        cc.bypass = False

        print(sd.query_devices())
        pr = audio.PlayRec(rec, ml_conf, [cc])
        pr.start()
        # Add parameterchange
        # 1. Bounds for entire playing surface:
        b = actions.Bounds(phi=[0, 360])
        pm = actions.ParameterMapper.from_bounds(
            b, cc, "phi", ["gain", "treble"]
        )
        pc = actions.ParameterChange([b], cc, [pm])
        pr.actions.append(pc)
        b = actions.Bounds(phi=[0, 360])
        pm = actions.ParameterMapper.from_bounds(
            b, cc, "phi", ["level"], lambda x: 1 / x
        )
        pc = actions.ParameterChange([b], cc, [pm])
        pr.actions.append(pc)

        plan_thread = threading.Thread(target=plan_callback, args=(pr,))
        plan_thread.start()
        # ap.join()
        ap2.join()
        plan_thread.join()
        sd.sleep(10)
