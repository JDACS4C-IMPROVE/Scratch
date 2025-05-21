#!/usr/bin/env python

import sys

if len(sys.argv) != 2:
    print("provide directory!")
    exit(1)

directory = sys.argv[1]

import pandas as pd

df_ge = pd.read_parquet(directory + "/ge_train_data.parquet")
print("ge: %i" % len(df_ge))

print(str(df_ge))

df_md = pd.read_parquet(directory + "/md_train_data.parquet")
print("md: %i" % len(df_md))
print(str(df_md))

df_rsp = pd.read_parquet(directory + "/rsp_train_data.parquet")
print("rsp: %i" % len(df_rsp))
print(str(df_rsp))
