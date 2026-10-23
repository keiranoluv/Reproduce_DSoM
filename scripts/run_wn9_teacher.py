#!/usr/bin/env python3
"""Independent WN9 teacher stage; default plan, user executes smoke/full."""

from wn9_workflow import parser, run_stage

if __name__ == "__main__":
    run_stage("teacher", parser("teacher").parse_args())
