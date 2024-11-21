#!/usr/bin/env python

import sys
import pandas as pd

filenames = sys.argv[1:]

for filename in filenames:
    try:
        df_rsp = pd.read_parquet(filename)
    except Exception as e:
        print("show-parquet.py: ERROR")
        print(str(e))
        exit(1)

    prefix = filename + ":"
    print("%-24s %5i" % (prefix, len(df_rsp)))
