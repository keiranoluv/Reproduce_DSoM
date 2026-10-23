#!/usr/bin/env python3
"""Check installed dependency metadata; CUDA checks require explicit --gpu."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", action="store_true")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Explicitly initialize CUDA and run a tiny tensor operation",
    )
    args = p.parse_args()
    requirements = ["torch==2.7.0", "torchvision==0.22.0"]
    files = (
        ["train.txt"]
        + (["data-features.txt"] if args.features else [])
        + (["dev.txt"] if args.dev else [])
        + (["tracking.txt"] if args.wandb else [])
    )
    for name in files:
        requirements.extend(
            line
            for line in (ROOT / "requirements" / name).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    errors, installed = [], {}
    if sys.version_info[:2] != (3, 12):
        errors.append("Documented profile requires Python 3.12")
    for requirement in dict.fromkeys(requirements):
        name, wanted = requirement.split("==")
        try:
            actual = importlib.metadata.version(name)
            installed[name] = actual
            if actual.split("+")[0] != wanted:
                errors.append(f"{name}: expected {wanted}, installed {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {name}")
    report = dict(
        python=platform.python_version(),
        executable=sys.executable,
        installed=installed,
        cuda_checked=False,
        errors=errors,
    )
    if args.gpu:
        import torch

        report["cuda_checked"] = True
        if not torch.cuda.is_available():
            errors.append("CUDA unavailable")
        else:
            value = (torch.ones(1, device="cuda") + 1).item()
            report.update(
                gpu=torch.cuda.get_device_name(0),
                torch_cuda=torch.version.cuda,
                tensor_check=value == 2.0,
            )
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
