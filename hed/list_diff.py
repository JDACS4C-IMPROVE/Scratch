#!/usr/bin/env python

import csv

# For comma-separated numbers:
import locale
locale.setlocale(locale.LC_ALL, "")


def main():
    args = parse_args()
    table_true, table_pred = read_tables(args)
    diff_sum, diff_sum2 = report_diff(args, table_true, table_pred)
    print("")
    print(f"diff_sum:  {diff_sum:8n}")
    print(f"diff_sum2: {diff_sum2:8n}" )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(prog="list_diff")
    parser.add_argument("N", type=int,
                        help="Only report top-N from table_true")
    parser.add_argument("input_csv")
    args = parser.parse_args()
    return args


def read_tables(args):
    table_true = []
    table_pred = []
    with open(args.input_csv, "r") as fp:
        reader = csv.reader(fp)
        next(reader) # header
        next(reader) # header
        for row in reader:
            table_true.append(row[1])
            table_pred.append(row[4])
    return table_true, table_pred


def report_diff(args, table_true, table_pred):
    index_true = 0
    diff_sum   = 0
    diff_sum2  = 0
    for i in range(0, args.N):
        key = table_true[i]
        index_pred = table_find(table_pred, key)
        diff = index_pred - index_true
        print("%4i %s %s %4i" %
              (index_true, key, table_pred[i], diff))
        diff_sum   += diff
        diff_sum2  += diff ** 2
        index_true += 1
    return diff_sum, diff_sum2


def table_find(table, key):
    index = 0
    for entry in table:
        # print(" %s %s" % (key, entry))
        if key == entry:
            return index
        index += 1
    raise Exception("not found: key=" + key)


if __name__ == "__main__": main()
