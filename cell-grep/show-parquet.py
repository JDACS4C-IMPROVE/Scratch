#!/usr/bin/env python

import sys
import pandas as pd

def msg(text):
    print("show-parquet.py: " + text)

if len(sys.argv) != 2:
    msg("provide a filename!")
    exit(1)
    
filename = sys.argv[1]

try:
    df_rsp = pd.read_parquet(filename)
except Exception as e:
    msg("ERROR")
    print(str(e))
    exit(1)

print(str(df_rsp))
