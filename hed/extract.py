#!/usr/bin/env python

"""
EXTRACT PY
Extract test metric scores from logs by index into CSV
Creates directory/scores.csv
"""

def main():
    args = parse_args()
    # print(str(args))
    extract(args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=
                                     "Extract scores from experiment")
    parser.add_argument("directory",
                        help="The experiment directory")
    parser.add_argument("metric",
                        help="The error metric")
    args = parser.parse_args()
    return args


def extract(args):
    output = args.directory + "/scores.csv"
    with open(output, "w") as fp_out:
        print("writing to: " + output)
        extract_all(args, fp_out)


def extract_all(args, fp_out):
    import os
    for index in range(0, 1000):
        run = "RUN%03i" % index
        scores_json = f"{args.directory}/run/{run}/test_scores.json"
        if os.path.exists(scores_json):
            get_error(args, fp_out, index, scores_json)
        else:
            print("done: extracted %i scores." % index)
            break


def get_error(args, fp_out, index, scores_json):
    import json
    with open(scores_json, "r") as fp_in:
        J = json.load(fp_in)
    if args.metric not in J:
        exit("metric not found: '%s'" % args.metric)
    fp_out.write("%03i,%.9f\n" % (index, J[args.metric]))


if __name__ == "__main__": main()
