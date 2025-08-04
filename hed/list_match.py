#!/usr/bin/env python

"""
LIST MATCH PY
Reports the recall
"""

import sys

def main():
    args = parse_args()
    hits, total = count_matches(args)
    ratio = hits / total
    print("hits: %4.1f%%" % (100 * ratio))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(prog="list_match")
    parser.add_argument("file_true")
    parser.add_argument("file_pred")
    args = parser.parse_args()
    return args


def count_matches(args):
    runs_true = {}
    hits  = 0
    total = 0
    with open(args.file_true, "r") as fp:
        for run in fp:
            total += 1
            runs_true[run] = None
    with open(args.file_pred, "r") as fp:
        for run in fp:
            if run in runs_true:
                print("hit: " + run)
                hits += 1
    return hits, total


if __name__ == "__main__": main()
