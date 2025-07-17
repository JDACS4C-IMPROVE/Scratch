#!/usr/bin/env python

"""
GEN JSONs
Make JSON fragments and put them in the UPF
"""


import json


def main():
    args = parse_args()
    jsons = []
    for i in range(0, args.count):
        J = make_json(args, i)
        jsons.append(J)
    write(jsons, args.output)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("count", type=int,
                        help="number of runs to make")
    parser.add_argument("output", type=str,
                        help="output UPF file")
    args = parser.parse_args()
    return args


def make_json(args, i):
    J = {"id": "RUN%03i" % i,
         "index": i}
    J = json.dumps(J)
    return J


def write(jsons, output):
    count = len(jsons)
    with open(output, "w") as fp:
        fp.write("# GEN JSONS\n")
        fp.write("# COUNT:     %i\n" % count)
        fp.write("# TIMESTAMP: " + timestamp() + "\n")
        fp.write("")
        for J in jsons:
            fp.write(J)
            fp.write("\n")
    print("wrote %i JSONs to " % count + output)


def timestamp():
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__": main()
