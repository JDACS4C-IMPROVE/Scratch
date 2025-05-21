
def main():
    args = parse_args()
    dfs = load_dataframes_pd(args.infiles)
    df = concat(args, dfs)
    store_dataframe_pd(args.outfile, df)
    log("DONE.")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("outfile", type=str,
                        help="name of output file")
    parser.add_argument("infiles", type=str, action="append",
                        help="name of input file (multiple)")
    args = parser.parse_args()
    return args


def load_dataframes_pd(infiles):
