
import time


program_start = time.time()


def log(txt="HERE", last_time=None):
    """
    Log message txt.
    Can be used to calculate time intervals with symbol '+',
    simply provide the start timestamp in last_time,
    else uses program start time and symbol is blank.
    """
    import datetime
    global program_start
    if last_time is None:
        last_time = program_start
        symbol = " "
    else:
        symbol = "+"
    now = time.time()
    dt = datetime.datetime.now()
    # Milliseconds:
    ms = dt.strftime("%f")[0:3]
    ds = dt.strftime("%Y-%m-%d %H:%M:%S.") + ms
    print("%s %s %9.3f %s" % (ds, symbol, now - last_time, txt))
    return time.time()


def log_start():
    """ Reset the start time """
    global program_start
    program_start = time.time()


def read_df_pq(infile):
    """ Read DataFrame ParQuet """
    # timestamp = log("importing pandas start")
    import pandas as pd
    # log("importing pandas stop", timestamp)
    timestamp = log("reading: " + infile)
    df = pd.read_parquet(infile)
    log("read:    OK", timestamp)
    return df


def write_df_pq(outfile, df):
    """ Write DataFrame ParQuet """
    import os
    timestamp = log("writing: " + outfile)
    if os.path.exists(outfile):
        raise(FileExistsError("refusing to overwrite: " + outfile))
    D = os.path.dirname(outfile)
    os.makedirs(D, exist_ok=True)
    df.to_parquet(outfile)
    log("write:   OK", timestamp)
