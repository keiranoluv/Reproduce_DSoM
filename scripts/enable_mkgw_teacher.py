#!/usr/bin/env python3
"""Minimal safe patch: add MKG-W to Teacher dataset choices only."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src/learn.py"
text = PATH.read_text()
old = 'big_datasets = ["WN9", "WN18", "FB15K-237", "MKG-Y"]'
new = 'big_datasets = ["WN9", "WN18", "FB15K-237", "MKG-Y", "MKG-W"]'
if new in text:
    print("Teacher MKG-W dataset choice: already enabled")
elif text.count(old) == 1:
    PATH.write_text(text.replace(old, new, 1))
    print("Teacher MKG-W dataset choice: enabled")
else:
    raise RuntimeError("Unexpected src/learn.py dataset choices; edit manually")
for filename in ("src/learn.py", "src/kd_learn.py"):
    value = (ROOT / filename).read_text()
    for arg in ("--git_commit", "--experiment_id"):
        if arg not in value:
            raise RuntimeError(f"Missing {arg} in {filename}")
