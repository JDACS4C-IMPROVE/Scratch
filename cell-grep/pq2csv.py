#!/usr/bin/env python

import sys

def msg(text):
    print("pq2csv.py: " + text)

if len(sys.argv) != 3:
    msg("provide PARQUET CSV")
    exit(1)
    
file_pqt = sys.argv[1]
file_csv = sys.argv[2]

try:
    import pandas as pd
    df = pd.read_parquet(file_pqt)
    df.to_csv(file_csv)
except Exception as e:
    msg("ERROR")
    print(str(e))
    exit(1)
