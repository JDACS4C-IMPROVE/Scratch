#!/usr/bin/env python


"""
STAT PARQUET PY

Pipe to column -t | sort -n -k 4
"""

import datetime
import os
import sys
import pandas as pd

filenames = sys.argv[1:]

for filename in filenames:

    print(filename, end=" ")
    try:
        s = os.stat(filename)
        print(format(s.st_size, ",.0f"), end=" ")
        dt = datetime.datetime.fromtimestamp(s.st_mtime)
        print(dt.strftime("%Y-%m-%d %H:%M:%S"), end=" ")
        df_rsp = pd.read_parquet(filename)
    except Exception as e:
        print("show-parquet.py: ERROR for file: '%s'" % filename)
        print(str(e))
        exit(1)

    print("%i" % len(df_rsp))
