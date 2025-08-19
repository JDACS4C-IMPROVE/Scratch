#!/usr/bin/env python

"""
CAT PARQUETS
"""

from utils import read_df_pq, write_df_pq


def main():
    timestamp = log("start...")
    args = parse_args()
    # print(str(args))
    try:
        dfs = read_dfs_pq(args.infiles[0])
    except FileNotFoundError as e:
        abort(str(e))
    log("read_dfs: count: %i" % len(dfs), timestamp)
    import pandas as pd
    df = pd.concat(dfs)
    try:
        write_df_pq(args.outfile, df)
    except FileExistsError as e:
        abort(str(e))
    log("done.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("outfile",
                        help="name of output file")
    # This list has unwanted extra nesting:
    parser.add_argument("infiles", nargs="+", action="append",
                        help="name of input file (multiple)")
    args = parser.parse_args()
    return args


def read_dfs_pq(infiles):
    import time
    result = []
    timestamp = time.time()
    for infile in infiles:
        timestamp = log("open: " + infile)
        df = read_df_pq(infile)
        log("read: " + infile, timestamp)
        result.append(df)
    return result


def concat(dfs):
    result = dfs[0]
    log("length 0: %i" % len(df_rsp))

    for i, df in dfs[1:]:
        result.concat(df)
        log("length %i: %i" % (i, len(df_rsp)))
    return result


def log(txt, last_time=None):
    import utils
    timestamp = utils.log("cat_parquets: " + txt, last_time)
    return timestamp


def abort(txt, last_time=None):
    log("ABORT: " + txt, last_time)
    exit(1)


if __name__ == "__main__": main()
