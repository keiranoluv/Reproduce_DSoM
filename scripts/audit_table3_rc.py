#!/usr/bin/env python3
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "R1 Shared Adagrad": ROOT
    / "runs/student/FB15K-237/table3_batch1024/full/train.log",
    "R2 Separate Adam": ROOT
    / "runs/student/FB15K-237/table3_separate_adam/full/train.log",
}

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

    if not path.exists():
        print("Log not found:", path)
        return

    text = path.read_text(errors="replace")

    policies = []
    for m in PATTERN.finditer(text):
        actions = [int(x.strip()) for x in m.group(6).split(",")]
        policies.append(
            {
                "batch": int(m.group(1)),
                "entropy": float(m.group(2)),
                "max_prob": float(m.group(3)),
                "near_det": int(m.group(4)) / int(m.group(5)),
                "actions": actions,
            }
        )

    rc = []
    for m in RC_PATTERN.finditer(text):
        rc.append(
            {
                "batch": int(m.group(1)),
                "adv_nonzero": int(m.group(3)),
                "grad_norm": float(m.group(5)),
                "weight_change": float(m.group(6)),
            }
        )

    mlps = []
    for m in MLP_PATTERN.finditer(text):
        mlps.append(
            {
                "batch": int(m.group(1)),
                "hidden_rms": float(m.group(2)),
                "logit_absmax": float(m.group(3)),
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
        print("  Entropy mean:", round(mean(x["entropy"] for x in records), 6))
        print("  Max prob mean:", round(mean(x["max_prob"] for x in records), 6))
        print(
            "  Near deterministic mean:", round(mean(x["near_det"] for x in records), 6)
        )

        totals = [sum(x["actions"][i] for x in records) for i in range(7)]
        total = sum(totals)

        print("  Strategy proportions:", [round(x / total, 5) for x in totals])

    first = policies[:20]
    last = policies[-20:]

    summarize("First 20 logged batches", first)
    summarize("Last 20 logged batches", last)

    collapsed = [x for x in policies if x["entropy"] < 0.01 and x["max_prob"] > 0.99]

    if collapsed:
        print("\nFirst near-collapse observation:", collapsed[0]["batch"])
    else:
        print("\nNo near-collapse observation under threshold")

    if rc:
        print("\nRC gradient statistics")
        print("  Mean grad norm:", mean(x["grad_norm"] for x in rc))
        print(
            "  Zero-grad logged fraction:",
            sum(x["grad_norm"] == 0 for x in rc) / len(rc),
        )
        print("  Mean weight change:", mean(x["weight_change"] for x in rc))

    if mlps:
        print("\nMLP statistics")
        print("  First hidden RMS:", mlps[0]["hidden_rms"])
        print("  Last hidden RMS:", mlps[-1]["hidden_rms"])
        print("  First logit absmax:", mlps[0]["logit_absmax"])
        print("  Last logit absmax:", mlps[-1]["logit_absmax"])

    print("\nFinal logged policy:")
    print(policies[-1])


for name, path in RUNS.items():
    analyze(name, path)
