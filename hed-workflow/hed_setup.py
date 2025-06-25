
# HED SETUP

import os, sys

from runner_utils import ModelResult

import partition_uno_pq as pupq


def get_marker(params, create_dir=False):
    index = int(params["index"])
    D = params["instance_directory"]
    markers = D + "../../../markers"
    markers = os.path.realpath(markers)
    if create_dir:
        os.makedirs(markers, exist_ok=True)
    filename = markers + "/marker-%03i.txt" % index
    msg("marker: " + filename)
    return filename


def pre_run(params):
    """
    Check if run is already complete
    Do the data slicing for a leave-one-drug-out case
    Resets params[input_dir]  !
    Resets params[output_dir] !
    """
    # print("hed_setup:pre: " + str(params))
    orig_dir = params["input_dir"]
    infile = orig_dir + "/rsp_train_data.parquet"
    filename = get_marker(params)
    b = os.path.exists(filename)
    if b:
        print("hed_setup:pre: already DONE")
        return ModelResult.SKIP
    index = int(params["index"])
    input_dir = orig_dir + "/../index-%03i" % index
    input_dir = os.path.realpath(input_dir)
    params["input_dir" ] = input_dir
    params["output_dir"] = params["instance_directory"]
    pattern = input_dir + "/rsp_@@@_data"
    from types import SimpleNamespace
    cfg = SimpleNamespace(partition = "by_drug",
                          index     = index,
                          infile    = infile,
                          out       = pattern)
    # msg("partition: " + str(cfg))
    pupq.run(cfg)
    for data in [ "ge", "md", "rsp" ]:
        for stage in [ "train", "val", "test" ]:
            name = "%s_%s_data.parquet" % (data, stage)
            # msg("name: " + name)
            if os.path.exists(input_dir + "/" + name):
                # msg("  exists.")
                continue
            # msg("  linking.")
            os.link(orig_dir  + "/" + name,
                    input_dir + "/" + name)
    return ModelResult.SUCCESS


def post_run(params, output_map):
    """
    Mark this run as complete
    """
    print("hed_setup:post: " + str(params))
    filename = get_marker(params, create_dir=True)
    with open(filename, "w") as fp:
        fp.write("DONE\n")
    input_dir =  params["input_dir"]
    import glob
    L = glob.glob(input_dir + "/*.parquet")
    for name in L:
        # msg("unlink: " + name)
        os.unlink(name)


def msg(s):
    print("hed_setup: " + s)
    sys.stdout.flush()
