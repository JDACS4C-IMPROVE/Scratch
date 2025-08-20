
"""
HED SETUP

Contains pre/post methods for model_runner,
called before/after model run.
Specified by hyperparams "pre_module"/"post_module"
This module has its own logger (see log())
based on the Supervisor log_tools features.

Can remove markers: model_runner's stop.marker works!
"""

import os, sys

# Supervisor functionality:
from runner_utils import ModelResult
from log_tools import *
from sv_utils import fail

import partition_uno_pq as pupq


logger = None


def get_marker(params, create_dir=False):
    index = int(params["index"])
    D = params["instance_directory"]
    markers = D + "../../../markers"
    markers = os.path.realpath(markers)
    if create_dir:
        os.makedirs(markers, exist_ok=True)
    filename = markers + "/marker-%03i.txt" % index
    logger.debug("marker: " + filename)
    return filename


def pre_run(params):
    """
    Check if run is already complete
    Do the data slicing for a leave-one-drug-out case
    Modifies params[input_dir]  !
    Modifies params[output_dir] !
    """
    log("pre: " + str(params))

    # Basic parameters:
    orig_dir = params["input_dir"]

    if check_marker(params):
        return ModelResult.SKIP

    # Make a unique input dir for this run:
    index = int(params["index"])
    input_dir = orig_dir + "/../index-%03i" % index
    input_dir = os.path.realpath(input_dir)
    params["input_dir" ] = input_dir
    params["output_dir"] = params["instance_directory"]

    cfg_xpu()
    do_partition(orig_dir, input_dir, index)
    link_inputs (orig_dir, input_dir)

    # Copy out generated inputs for debugging:
    # import shutil
    # shutil.copytree(input_dir, params["instance_directory"] + "/inputs")

    return ModelResult.SUCCESS

def check_marker(params):
    """ Check if this run was already done by a prior workflow """

    filename = get_marker(params)
    b = os.path.exists(filename)
    if b:
        print("hed_setup:pre: already DONE")
    return b

def cfg_xpu():
    """ Aurora only.  Setup TensorFlow for single XPU per rank """

    rank_local = int(os.environ.get('PALS_LOCAL_RANKID', '0'))
    log("cfg_xpu(): local rank: %i" % rank_local)

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("XPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[rank_local], "XPU")


def do_partition(orig_dir, input_dir, index):
    """ Construct the arguments for the partitioner and run """

    infile = orig_dir + "/rsp_merged.parquet"
    pattern = input_dir + "/rsp_@@@_data"
    from types import SimpleNamespace
    cfg = SimpleNamespace(partition = "by_drug",
                          index     = index,
                          infile    = infile,
                          out       = pattern)
    # Do the partition!
    pupq.run(cfg)


def link_inputs(orig_dir, input_dir):
    """ Link other data into input dir w/o a full file copy """

    for label in [ "ge", "md"]:
        name = "%s_merged_data.parquet" % label
        log("link: name: " + name)
        if os.path.exists(input_dir + "/" + name):
            log("  link: exists.")
            continue
        log("link: linking.")
        if not os.path.exists(orig_dir  + "/" + name):
            fail("link target does not exist: " +
                 orig_dir  + "/" + name)
        os.link(orig_dir  + "/" + name,
                input_dir + "/" + name)


def post_run(params, output_map):
    """ Mark this run as complete and clean up """
    log("post: " + str(params))
    filename = get_marker(params, create_dir=True)
    with open(filename, "w") as fp:
        fp.write("DONE\n")
    input_dir =  params["input_dir"]
    import glob
    L = glob.glob(input_dir + "/*.parquet")
    for name in L:
        # logger.debug("unlink: " + name)
        os.unlink(name)


def log(s):
    global logger
    logger = get_logger(logger, "hed_setup")
    logger.info(s)
    sys.stdout.flush()


def debug(s):
    global logger
    logger = get_logger(logger, "hed_setup")
    logger.debug(s)
    sys.stdout.flush()
