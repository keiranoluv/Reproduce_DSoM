#!/usr/bin/env python3
"""Independent WN9 student stage, consuming a validated completed teacher run."""

from wn9_workflow import parser, run_stage

if __name__ == "__main__":
    run_stage("student", parser("student").parse_args())
