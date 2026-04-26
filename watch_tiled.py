"""
Tiled watcher — runs N envs with a loaded policy and displays them in a grid.

Usage:
    python watch_tiled.py                                            # loads checkpoints/hexapod_stage1_test/final
    python watch_tiled.py checkpoints/hexapod_stage1_test/final      # explicit path
    python watch_tiled.py checkpoints/hexapod_stage1_test/final 9    # 9 envs (3x3)

Press Q to quit.
"""
import sys
import math
import numpy as np
import cv2
from stable_baselines3 import PPO

from envs.hexapod_env import HexapodEnv

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hexapod_stage1_test/final"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 16

COLS     = math.ceil(math.sqrt(N))
ROWS     = math.ceil(N / COLS)
FRAME_W  = 320
FRAME_H  = 240

print(f"Loading: {ckpt}")
print(f"Envs: {N} ({COLS}x{ROWS} grid) | Frame: {COLS*FRAME_W}x{ROWS*FRAME_H}")

envs = [HexapodEnv(stage=1, render_mode="rgb_array") for _ in range(N)]

model = PPO.load(ckpt, env=envs[0])

obs        = [e.reset()[0] for e in envs]
ep_rewards = [0.0] * N
ep_steps   = [0]   * N

cv2.namedWindow(f"Training — {N} Hexapods", cv2.WINDOW_NORMAL)

while True:
    frames = []

    for i, e in enumerate(envs):
        action, _ = model.predict(obs[i], deterministic=True)
        obs[i], rew, term, trunc, _ = e.step(action)
        ep_rewards[i] += rew
        ep_steps[i]   += 1

        if term or trunc:
            obs[i], _ = e.reset()
            ep_rewards[i] = 0.0
            ep_steps[i]   = 0

        frame = e.render()
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cmd = e._cmd
        label = f"#{i}  cmd({cmd[0]:+.2f},{cmd[1]:+.2f})"
        cv2.putText(frame, label, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        frames.append(frame)

    while len(frames) < COLS * ROWS:
        frames.append(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))

    rows = [np.hstack(frames[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    grid = np.vstack(rows)

    cv2.imshow(f"Training — {N} Hexapods", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
for e in envs:
    e.close()
