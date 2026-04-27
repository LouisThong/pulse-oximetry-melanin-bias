#!/usr/bin/env python3
"""
Capture CSV output from the MAX30101 sketch and save it to a file.

By default the file is auto-named P1.csv, P2.csv, P3.csv ... in ~/Desktop
(the script picks the next free number each run).

Usage:
    python3 capture_ppg.py                                    # saves next P#.csv on Desktop
    python3 capture_ppg.py --port /dev/cu.usbserial-1420      # custom port
    python3 capture_ppg.py --dir ~/Documents --prefix run     # saves run1.csv, run2.csv ...
    python3 capture_ppg.py --out ~/Desktop/custom_name.csv    # override the auto-name
"""

import argparse
import re
import sys
import time
from pathlib import Path

import serial  # pip3 install pyserial


def next_auto_path(dir_path: Path, prefix: str) -> Path:
    """Return dir_path / <prefix><N>.csv where N is the next unused integer."""
    dir_path.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.csv$", re.IGNORECASE)
    used = [
        int(m.group(1))
        for f in dir_path.iterdir()
        if f.is_file() and (m := pattern.match(f.name))
    ]
    n = max(used) + 1 if used else 1
    return dir_path / f"{prefix}{n}.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/cu.usbserial-1420")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument(
        "--dir", default="~/Desktop",
        help="Folder where auto-named P#.csv files go (default: ~/Desktop).",
    )
    p.add_argument(
        "--prefix", default="P",
        help="Filename prefix for auto-naming (default: P -> P1.csv, P2.csv ...).",
    )
    p.add_argument(
        "--out", default=None,
        help="Explicit output path. Overrides --dir/--prefix auto-naming.",
    )
    p.add_argument(
        "--timeout", type=float, default=135.0,
        help="Hard cap in seconds (sketch runs ~120s).",
    )
    args = p.parse_args()

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = next_auto_path(Path(args.dir).expanduser().resolve(), args.prefix)

    print(f"Opening {args.port} @ {args.baud} baud ...")
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        # Many USB-serial adapters reset the MCU when the port opens.
        time.sleep(2.0)
        ser.reset_input_buffer()

        print(f"Writing to {out_path}")
        t0 = time.time()
        rows = 0
        with open(out_path, "w", buffering=1) as f:
            while True:
                if time.time() - t0 > args.timeout:
                    print("Timeout reached, stopping.")
                    break

                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("# capture complete"):
                    print("Sketch signalled end of capture.")
                    break

                f.write(line + "\n")
                rows += 1
                if rows % 100 == 0:
                    print(f"  {rows} lines... ({line})")

    print(f"Done. Saved {rows} lines to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"Serial error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
