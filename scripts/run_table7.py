#!/usr/bin/env python3
"""WN9 Table 7 E2E: fresh teacher -> fresh student -> raw/filtered checkpoint eval."""

from wn9_workflow import e2e, parser

if __name__ == "__main__":
    e2e(parser("e2e").parse_args())
