"""tools/web_controller_test.py — drive the bot's web controller from a script and measure it.

    python tools/web_controller_test.py --host 192.168.1.42 [--port /dev/cu.usbmodem101] [--dry]

Speaks the same WebSocket protocol as the page (wifi_web.ino / web_page.h): "S lx ly rx ry sl sr
up dn wd nr spd" at 20 Hz, "C stand|rest|stop", "P" pings. Runs a fixed sequence that exercises
single and COMBINED inputs, prints round-trip latency, the bot's telemetry and (with --port) the
bot's serial log lines, so the whole path phone -> WiFi -> policy -> servos can be checked
without touching the bot. --dry only pings and reads telemetry (no motion commands).
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import threading
import time
import urllib.request

import websockets

SEQ = [  # (seconds, lx, ly, rx, ry, sl, sr, up, dn, wd, nr, label)
    (2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "idle (keepalive only)"),
    (3.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, "stick forward, full"),
    (2.0, 0.7, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, "stick forward-right, 70 %"),
    (2.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, "spin left button alone"),
    (3.0, 0, 1.0, 0, 0, 1, 0, 0, 0, 0, 0, "stick forward + spin left  (COMBINED -> arc)"),
    (1.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, "hold RAISE 1.5 s (-> 30 mm)"),
    (2.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, "stick forward while raised"),
    (1.5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, "hold LOWER 1.5 s"),
    (1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, "hold WIDER 1 s"),
    (2.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, "tilt stick right (roll)"),
    (2.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, "tilt stick up (pitch)"),
    (2.0, 0, 0.8, 0.5, 0.5, 0, 1, 0, 0, 0, 0, "EVERYTHING: walk + spin right + tilt"),
    (1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "release all"),
]


def serial_reader(port: str, stop: threading.Event) -> None:
    import serial
    s = serial.Serial(port, 115200, timeout=0.1)
    while not stop.is_set():
        line = s.readline().decode(errors="replace").strip()
        if line:
            print(f"      bot| {line}", flush=True)
    s.close()


async def run(host: str, dry: bool) -> None:
    page = urllib.request.urlopen(f"http://{host}/", timeout=5).read()
    print(f"GET / -> {len(page)} bytes, title ok: {b'<title>hexapod</title>' in page}")
    async with websockets.connect(f"ws://{host}/ws", open_timeout=5) as ws:
        print("websocket open")
        tele, rtts, pending = {}, [], {}

        async def reader():
            async for m in ws:
                if m.startswith("T "):
                    f = m.split()
                    tele.update(v=f[1:4], h=f[4], w=f[5], pitch=f[6], roll=f[7], gain=f[8], state=f[9], tick=f[10], net=f[11], late=f[13] if len(f) > 13 else "?")
                elif m.startswith("M "):
                    print(f"      msg| {m[2:]}", flush=True)
                elif m.startswith("P"):
                    k = m[1:].strip() or "p"
                    if k in pending:
                        rtts.append((time.perf_counter() - pending.pop(k)) * 1000)

        rt = asyncio.create_task(reader())

        async def ping(i):
            pending[str(i)] = time.perf_counter(); await ws.send(f"P{i}")

        for i in range(10):
            await ping(i); await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)
        print(f"ping RTT ms: median {statistics.median(rtts):.1f}  max {max(rtts):.1f}  ({len(rtts)}/10 answered)")
        if dry:
            await asyncio.sleep(1.0); print("telemetry:", tele); rt.cancel(); return

        await ws.send("C stand"); await asyncio.sleep(1.0)
        for dur, lx, ly, rx, ry, sl, sr, up, dn, wd, nr, label in SEQ:
            t_cmd = time.perf_counter(); first_change = None; v0 = tuple(tele.get("v", ()))
            print(f"\n>>> {label}")
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < dur:
                await ws.send(f"S {lx:.3f} {ly:.3f} {rx:.3f} {ry:.3f} {sl} {sr} {up} {dn} {wd} {nr} 0.85")
                if first_change is None and tuple(tele.get("v", ())) != v0:
                    first_change = (time.perf_counter() - t_cmd) * 1000
                await asyncio.sleep(0.05)
            v = tele.get("v", "?")
            print(f"    after {dur:.1f}s: vel {v}  height {tele.get('h')} mm  width {tele.get('w')} mm  "
                  f"pitch/roll {tele.get('pitch')}/{tele.get('roll')}°  gain {tele.get('gain')}  state {tele.get('state')}  "
                  f"tick avg {tele.get('tick')} ms  net avg {tele.get('net')} ms  late {tele.get('late')}"
                  + (f"  | telemetry reacted after {first_change:.0f} ms" if first_change else ""))
        await ws.send("C rest"); await asyncio.sleep(3.0)
        print("\nfinal:", tele)
        rt.cancel()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", default=None, help="serial port to mirror the bot's log")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    stop = threading.Event()
    if args.port:
        threading.Thread(target=serial_reader, args=(args.port, stop), daemon=True).start()
    try:
        asyncio.run(run(args.host, args.dry))
    finally:
        stop.set(); time.sleep(0.3)
    sys.exit(0)


if __name__ == "__main__":
    main()
