#!/usr/bin/env python

"""
MK PARQUET
"""


import random

import utils


def main():
    timestamp = log("start...")
    args = parse_args()
    # print(str(args))
    df = mk_pq(args)
    log("df: count: %i" % len(df), timestamp)
    try:
        utils.write_df_pq(args.outfile, df)
    except FileExistsError as e:
        abort(str(e))
    log("done.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=
                                     "Make a test Parquet file")
    parser.add_argument("count", type=int,
                        help="number of rows to create")
    parser.add_argument("study",
                        help="study name to use")
    parser.add_argument("outfile",
                        help="name of output file")
    args = parser.parse_args()
    return args


def mk_pq(args):
    import pandas as pd
    samples = mk_samples(args.count)
    drugs   = mk_drugs  (args.count)
    aucs    = mk_aucs   (args.count)
    studies = mk_studies(args.count, args.study)

    df = pd.DataFrame({ "improve_sample_id": samples,
                        "improve_chem_id":   drugs,
                        "auc":               aucs,
                        "study":             studies })
    df.rename_axis("index")
    return df


def mk_samples(count):
    result = []
    for i in range(0, count):
        result.append(mk_sample())
    return result


def mk_sample():
    return "".join([random_char(),
                    random_char(),
                    "-",
                    random_digit(),
                    random_digit()])

def mk_drugs(count):
    result = []
    for i in range(0, count):
        result.append(mk_drug())
    return result


def mk_studies(count, study):
    result = []
    for i in range(0, count):
        result.append(study)
    return result


def mk_drug():
    return "Drug_" + str(random.randint(1, 2000))


def mk_aucs(count):
    result = []
    for i in range(0, count):
        result.append(random.random())
    return result


def random_char():
    c = random.randint(65, 91)
    return chr(c)


def random_digit():
    return str(random.randint(0, 9))


def log(txt, last_time=None):
    timestamp = utils.log("mk_pq: " + txt, last_time)
    return timestamp


def abort(txt, last_time=None):
    log("ABORT: " + txt, last_time)
    exit(1)


if __name__ == "__main__": main()
