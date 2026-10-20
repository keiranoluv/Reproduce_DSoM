#!/usr/bin/env python3
"""Audit RC policy behavior across Table 3 Student runs R1-R7."""

import argparse
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "r1": ("R1 Shared Adagrad", "table3_batch1024"),
    "r2": ("R2 Separate Adam", "table3_separate_adam"),
    "r3": ("R3 Scheduler MAX", "table3_r3_scheduler_max"),
    "r4": ("R4 Input Normalization", "table3_r4_input_norm"),
    "r5": ("R5 Batch 1000", "table3_r5_batch1000"),
    "r6": ("R6 Policy Temperature 2", "table3_r6_temp2"),
    "r7": ("R7 Continuous CE Reward", "table3_r7_continuous_ce"),
}

LOG_ROOT = ROOT / "runs/student/FB15K-237"

PATTERN = re.compile(
    r"POLICY batch=(\d+) "
    r"entropy=([-\d.eE+]+) "
    r"max_prob=([-\d.eE+]+) "
    r"near_deterministic=(\d+)/(\d+) "
    r"actions=\[([0-9,\s]+)\]"
)

RC_PATTERN = re.compile(
    r"RC batch=(\d+) "
    r"reward_mean=([-\d.eE+]+) "
    r"adv_nonzero=(\d+)/(\d+) "
    r"grad_norm=([-\d.eE+]+) "
    r"weight_change=([-\d.eE+]+)"
)

MLP_PATTERN = re.compile(
    r"MLP batch=(\d+).*?"
    r"hidden_rms=([-\d.eE+]+).*?"
    r"logit_absmax=([-\d.eE+]+)"
)


def analyze(name, path):
    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")

    if not path.is_file():
        print("Log not found:", path)
        return

    text = path.read_text(errors="replace")

    policies = []
    for match in PATTERN.finditer(text):
        actions = [int(x.strip()) for x in match.group(6).split(",")]

        if len(actions) != 7:
            raise ValueError(f"Expected 7 strategies, got {len(actions)}")

        policies.append(
            {
                "batch": int(match.group(1)),
                "entropy": float(match.group(2)),
                "max_prob": float(match.group(3)),
                "near_det": (int(match.group(4)) / int(match.group(5))),
                "actions": actions,
            }
        )

    rc = []
    for match in RC_PATTERN.finditer(text):
        rc.append(
            {
                "batch": int(match.group(1)),
                "adv_nonzero": int(match.group(3)),
                "grad_norm": float(match.group(5)),
                "weight_change": float(match.group(6)),
            }
        )

    mlps = []
    for match in MLP_PATTERN.finditer(text):
        mlps.append(
            {
                "batch": int(match.group(1)),
                "hidden_rms": float(match.group(2)),
                "logit_absmax": float(match.group(3)),
            }
        )

    if not policies:
        print("No POLICY records found")
        return

    print("Policy records:", len(policies))
    print("RC records:", len(rc))
    print("MLP records:", len(mlps))

    def summarize(label, records):
        if not records:
            return

        print(f"\n{label}")
        print(
            "  Entropy mean:",
            round(mean(x["entropy"] for x in records), 6),
        )
        print(
            "  Max prob mean:",
            round(mean(x["max_prob"] for x in records), 6),
        )
        print(
            "  Near deterministic mean:",
            round(mean(x["near_det"] for x in records), 6),
        )

        totals = [sum(x["actions"][i] for x in records) for i in range(7)]
        total = sum(totals)

        print(
            "  Strategy proportions:",
            [round(x / total, 5) for x in totals],
        )

    summarize(
        "First 20 logged batches",
        policies[:20],
    )
    summarize(
        "Last 20 logged batches",
        policies[-20:],
    )

    collapsed = [x for x in policies if x["entropy"] < 0.01 and x["max_prob"] > 0.99]

    if collapsed:
        print(
            "\nFirst near-collapse observation:",
            collapsed[0]["batch"],
        )
    else:
        print("\nNo near-collapse observation under threshold")

    if rc:
        print("\nRC gradient statistics")
        print(
            "  Mean grad norm:",
            mean(x["grad_norm"] for x in rc),
        )
        print(
            "  Zero-grad logged fraction:",
            sum(x["grad_norm"] == 0 for x in rc) / len(rc),
        )
        print(
            "  Mean weight change:",
            mean(x["weight_change"] for x in rc),
        )

    if mlps:
        print("\nMLP statistics")
        print("  First hidden RMS:", mlps[0]["hidden_rms"])
        print("  Last hidden RMS:", mlps[-1]["hidden_rms"])
        print(
            "  First logit absmax:",
            mlps[0]["logit_absmax"],
        )
        print(
            "  Last logit absmax:",
            mlps[-1]["logit_absmax"],
        )

    print("\nFinal logged policy:")
    print(policies[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Audit DSoM RC policy behavior for R1-R7."
    )

    parser.add_argument(
        "--run",
        choices=tuple(RUNS),
        help="Audit one specific run.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit all runs R1-R7.",
    )

    args = parser.parse_args()

    if args.run and args.all:
        parser.error("Choose either --run or --all")

    selected = [args.run] if args.run else list(RUNS)

    for key in selected:
        name, directory = RUNS[key]
        path = LOG_ROOT / directory / "full/train.log"
        analyze(name, path)


if __name__ == "__main__":
    main()
