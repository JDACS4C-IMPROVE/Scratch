#!/usr/bin/env python

import os
import sys
import pandas as pd


def help():
    print("""stat-parquet.py: usage:
             python stat-parquet.py [FILENAME]*""")


filenames = sys.argv[1:]

if len(filenames) == 0:
    help()
    exit(1)

for filename in filenames:

    if filename.startswith("-"):
        if filename == "-h":
            help()
            exit()
        else:
            print("show-parquet.py: ERROR")
            print("show-parquet.py: unknown flag: '%s'" % filename)
            exit(1)

    print(filename + ": ")
    try:
        s = os.stat(filename)
        bytes_s = f"{s.st_size:,}"
        print("\t bytes: %12s" % bytes_s)
        df_rsp = pd.read_parquet(filename)
    except Exception as e:
        print("show-parquet.py: ERROR")
        print(str(e))
        exit(1)

    print("\t rows:  %12i" % len(df_rsp))
