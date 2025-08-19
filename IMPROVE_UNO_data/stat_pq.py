#!/usr/bin/env python

import datetime
import os
import sys
import time


def main():
    args = parse_args()
    stat_files(args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="description")
    # This list has unwanted extra nesting:
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--drugs", action="store_true",
                        help="list drugs")
    parser.add_argument("-l", "--listing", action="store_true",
                        help="report as flat-format list")
    parser.add_argument("-t", "--time", action="store_true",
                        help="report timing")
    parser.add_argument("files", nargs="+", action="append",
                        help="name of input file (multiple)")
    args = parser.parse_args()
    if args.verbose: print("args: " + str(args))
    return args


def stat_files(args):

    import pandas as pd

    for filename in args.files[0]:

        if args.time: start = time.time()
        print(filename + ": ", end="")
        output_break(args)
        try:
            s = os.stat(filename)
            dt = datetime.datetime.fromtimestamp(s.st_mtime)
            print("date:", dt.strftime("%Y-%m-%d %H:%M:%S"), end="")
            output_break(args)
            bytes_s = f"{s.st_size:,}"
            print("bytes: %12s" % bytes_s, end="")
            output_break(args)
            df_rsp = pd.read_parquet(filename)
            rows = len(df_rsp)
            rows_s = f"{rows:,}"
            print("rows:  %12s" % rows_s)
            if args.verbose: print(str(df_rsp))
            if args.drugs:
                for index, row in df_rsp.iterrows():
                    print("drug: " + row["improve_chem_id"], end="")
                    output_break(args)
            print("")
        except Exception as e:
            print("stat-parquet.py: ERROR")
            print(str(e))
            exit(1)
    if args.time:
        stop = time.time()
        print("time: %0.3f" % (stop - start))


def output_break(args):
    # Insert a space or newline depending on output format
    if args.listing:
        print(" ", end="")
    else:
        print("\n\t", end="")


if __name__ == "__main__": main()
