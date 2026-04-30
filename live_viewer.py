"""
live_viewer.py — real-time mirror of env-0 from a running training session.

Connects to the shared-memory region that env-0 writes its qpos+qvel to every
step, and renders that state in a MuJoCo viewer at ~60 FPS. Decouples the
viewer process from the SubprocVecEnv worker so the same code path works
on both platforms (macOS's Cocoa viewer can only run under mjpython but
SubprocVecEnv workers can't be mjpython; SHM mediation sidesteps that).

Usage:
    macOS:  mjpython live_viewer.py
    Linux:  python   live_viewer.py

Workflow:
    1. Make sure train.py has WATCH_LIVE = True at the top.
    2. Launch training: `python train.py`  (env-0 creates the shared memory
       region the first time its step() is called).
    3. In a separate terminal: `mjpython live_viewer.py`.
    4. The viewer auto-connects to the running training and starts rendering.
       If you launch it before training, it'll wait until env-0 creates the
       region.

Either order is fine; the viewer retries until it finds the shared memory.
Quit the viewer at any time without affecting training.
"""

import os
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH    = "models/phantomx.xml"
LIVE_SHM_NAME = "hexapod_live_state"   # must match envs.hexapod_env.LIVE_SHM_NAME
TARGET_FPS    = 60.0


def wait_for_shm(name, model, timeout_seconds=None):
    """Block until the shared memory region appears. Returns SharedMemory + buffer."""
    nq, nv = model.nq, model.nv
    expected = (nq + nv + 1) * 8
    start = time.time()
    last_dot = 0.0
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name)
            if shm.size < expected:
                shm.close()
                raise RuntimeError(
                    f"shared memory '{name}' size {shm.size} smaller than expected "
                    f"{expected}; mismatched model? Aborting.")
            buf = np.ndarray((nq + nv + 1,), dtype=np.float64, buffer=shm.buf)
            return shm, buf
        except FileNotFoundError:
            now = time.time()
            if timeout_seconds is not None and (now - start) > timeout_seconds:
                raise TimeoutError(f"timed out waiting for shm '{name}'")
            if now - last_dot > 1.0:
                print(".", end="", flush=True)
                last_dot = now
            time.sleep(0.25)


def main():
    print(f"live_viewer — connecting to shared memory '{LIVE_SHM_NAME}'")
    print(f"  Make sure train.py has WATCH_LIVE = True and is running.")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    print(f"  waiting for shm", end="", flush=True)
    shm, buf = wait_for_shm(LIVE_SHM_NAME, model)
    print(f"\n  connected. Opening viewer at {TARGET_FPS:.0f} FPS.\n")

    nq, nv = model.nq, model.nv
    last_print = time.time()
    last_sim_t = -1.0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame_period = 1.0 / TARGET_FPS
            next_frame   = time.time()
            while viewer.is_running():
                # Snapshot current state.
                try:
                    sim_t = float(buf[nq + nv])
                    np.copyto(data.qpos, buf[:nq])
                    np.copyto(data.qvel, buf[nq:nq+nv])
                except Exception as e:
                    print(f"\n  shm read failed: {e}")
                    break

                mujoco.mj_forward(model, data)
                viewer.sync()

                # Print a periodic status line so user knows it's live.
                now = time.time()
                if now - last_print > 5.0:
                    rate = (sim_t - last_sim_t) / 5.0 if last_sim_t >= 0 else 0.0
                    print(f"  sim_t = {sim_t:8.2f}s   (sim rate ~{rate:.1f}× real time)")
                    last_print = now
                    last_sim_t = sim_t

                # Throttle to TARGET_FPS.
                next_frame += frame_period
                lag = next_frame - time.time()
                if lag > 0:
                    time.sleep(lag)
                else:
                    next_frame = time.time()
    finally:
        try:
            shm.close()
        except Exception:
            pass
        print("\n  live_viewer disconnected.")


if __name__ == "__main__":
    main()
