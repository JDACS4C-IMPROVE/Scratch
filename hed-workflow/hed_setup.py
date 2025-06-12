
# HED SETUP

import os, sys

from runner_utils import ModelResult

import partition_uno_pq as pupq


def get_marker(params, create_dir=False):
    index = int(params["index"])
    D = params["instance_directory"]
    markers = D + "/markers"
    if create_dir:
        os.make_dirs(markers, exist_ok=True)
    filename = markers + "/marker-%3i.txt" % index
    return filename


def pre_run(params):
    """
    Check if run is already complete
    Do the data slicing for a leave-one-drug-out case
    """
    print("hed_setup:pre: " + str(params))
    root = "/tmp/" + os.getenv("USER")
    index = int(params["index"])
    infile = root + "/rsp_train_data.parquet"
    filename = get_marker(params)
    b = os.path.exists(filename)
    if b:
        print("hed_setup:pre: already DONE")
        return ModelResult.SKIP
    pattern = root + "/index-%3i/rsp_@@@_data" % index
    cfg = {"partition" : "by_drug",
           "index"     : index,
           "infile"    : infile,
           "out"       : pattern
           }
    print("infile: " + infile)
    print("exists: %i" % os.path.exists(infile))
    sys.stdout.flush()
    pupq.run(cfg)
    return ModelResult.SUCCESS


def post_run(params):
    """
    Mark this run as complete
    """
    print("hed_setup:post: " + str(params))
    filename = get_marker(params, create_dir=True)
    with open(filename) as fp:
        fp.write("DONE\n")

