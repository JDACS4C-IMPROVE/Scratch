
"""
LIST DRUGS PY
"""


import partition_uno_pq as pupq
# Our utilities
import utils


def main():
    args = parse_args()
    try:
        run(args)
    except (IndexError, ValueError, FileExistsError) as e:
        abort(str(e))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("--index", type=int, default=None,
                        help="index to be used to print drug, " +
                        "omit for all")
    parser.add_argument("infile",
                        help="name of input file: RSP PARQUET")
    args = parser.parse_args()
    return args


def run(cfg):
    df = utils.read_df_pq(cfg.infile)
    if cfg.index is None:
        list_drugs(df)
    else:
        drug = pupq.select_drug(cfg.index, df=df)
        print_drug(cfg.index, drug)


def list_drugs(df):
    unique_drugs = df['improve_chem_id'].unique()
    total = len(unique_drugs)
    for index in range(0, total):
        drug = unique_drugs[index]
        print_drug(index, drug)


def print_drug(index, drug):
    print("%03i %s" % (index, drug))


if __name__ == "__main__": main()
