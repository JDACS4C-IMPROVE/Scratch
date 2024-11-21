#!/usr/bin/env python

import sys
import pandas as pd

try:
    df_rsp = pd.read_parquet(sys.argv[1])
except Exception as e:
    print("show-parquet.py: ERROR")
    print(str(e))
    exit(1)

print(str(df_rsp))
