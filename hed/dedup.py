#!/usr/bin/env python

"""
DEDUP.PY
Dedup run lines from stream
"""

import sys

runs = {}

for line in sys.stdin:
    run = line[3:6]
    if run in runs: continue
    runs[run] = None
    print(line, end="")
