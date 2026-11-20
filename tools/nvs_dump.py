"""tools/nvs_dump.py — decode an ESP32 NVS partition image (read-only backup path).

    esptool --chip esp32c3 --port <port> read_flash 0x9000 0x5000 nvs.bin
    python tools/nvs_dump.py nvs.bin [--json out.json]

Prints every live key as namespace/key = value. Used to recover the mini
hexapod's trims/signs from a board whose sketch has no export command.
Format: 4096-byte pages; 32-byte header, 32-byte entry-state bitmap, 126
entries of 32 bytes. Entry = ns(1) type(1) span(1) chunk(1) crc32(4) key(16)
data(8). Namespace index 0 entries define namespaces (data[0] = ns id).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

PAGE = 4096
ENTRY = 32
TYPES = {0x01: ("u8", "<B"), 0x11: ("i8", "<b"), 0x02: ("u16", "<H"), 0x12: ("i16", "<h"),
         0x04: ("u32", "<I"), 0x14: ("i32", "<i"), 0x08: ("u64", "<Q"), 0x18: ("i64", "<q"),
         0x21: ("str", None), 0x41: ("blob", None), 0x42: ("blob_data", None), 0x48: ("blob_idx", None)}


def decode(img: bytes) -> dict[str, dict[str, object]]:
    namespaces: dict[int, str] = {}
    raw: list[tuple[int, str, object]] = []
    for p in range(0, len(img), PAGE):
        page = img[p:p + PAGE]
        state = struct.unpack("<I", page[:4])[0]
        if state == 0xFFFFFFFF:          # erased page
            continue
        bitmap = page[32:64]
        i = 0
        while i < 126:
            st = (bitmap[i // 4] >> ((i % 4) * 2)) & 0x3
            e = page[64 + i * ENTRY: 64 + (i + 1) * ENTRY]
            if st != 0x2:                # 2 = written; 0 = erased, 3 = empty
                i += 1
                continue
            ns, typ, span = e[0], e[1], e[2]
            key = e[8:24].split(b"\0", 1)[0].decode(errors="replace")
            data = e[24:32]
            if ns == 0:
                namespaces[data[0]] = key
            elif typ in TYPES and TYPES[typ][1]:
                fmt = TYPES[typ][1]
                raw.append((ns, key, struct.unpack(fmt, data[:struct.calcsize(fmt)])[0]))
            elif typ == 0x21:
                size = struct.unpack("<H", data[:2])[0]
                payload = page[64 + (i + 1) * ENTRY: 64 + (i + 1) * ENTRY + size]
                raw.append((ns, key, payload.split(b"\0", 1)[0].decode(errors="replace")))
            i += max(1, span)
    out: dict[str, dict[str, object]] = {}
    for ns, key, val in raw:                      # later entries win (NVS keeps the newest live)
        out.setdefault(namespaces.get(ns, f"ns{ns}"), {})[key] = val
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    with open(a.image, "rb") as fh:
        data = decode(fh.read())
    for ns, kv in sorted(data.items()):
        for k, v in sorted(kv.items()):
            print(f"{ns}/{k} = {v}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(data, fh, indent=2)
        print("wrote", a.json, file=sys.stderr)


if __name__ == "__main__":
    main()
