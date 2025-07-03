
"""
PARTITION UNO PARQUET
"""

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# from collections import OrderedDict
# from sklearn.preprocessing import StandardScaler

import datetime
import time

from sklearn.model_selection import train_test_split

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
    parser.add_argument("partition", type=str,
                        help="partition type (random, by_cell, by_drug)")
    parser.add_argument("--index", type=int, default=None,
                        help="index to be used to delete cell or drug")
    parser.add_argument("infile",
                        help="name of input file")
    parser.add_argument("out",
                        help="pattern of output files: " +
                             "e.g., rsp_@@@_data w/o extension")
    args = parser.parse_args()
    log(f'using partition: {args.partition}')
    return args


# The input DataFrame- only read this once
df = None


def run(cfg):
    """
    The main interface function for main() or library calls
    cfg: argparse-formatted key-values
    Exceptions: IndexError, ValueError, FileExistsError
    """
    utils.log_start()
    log("run: %s %i" % (cfg.partition, cfg.index))
    global df
    if df is None:
        df = utils.read_df_pq(cfg.infile)
    train, val, test = delete_rows(cfg, df)
    write(cfg, train, val, test)
    log("done.")


def delete_rows(args, df):
    timestamp = time.time()
    if args.partition == "random":
        train, val, test = partition_random(df)
        timestamp = log(f'DONE_PARTITIONING_DATA {args.infile}', timestamp)

    elif args.partition == "by_cell":
        cell = select_cell(rank, df=df)
        timestamp = log("DONE_SELECTING_CELL", timestamp)
        print(cell)
        train, val, test = partition_cell(df, cell)
        timestamp = log(f'DONE_PARTITIONING_DATA {args.infile}', timestamp)

    elif args.partition == "by_drug":
        # drug: the drug name as a string
        drug = select_drug(args.index, df=df)
        timestamp = log("OK: " + drug, timestamp)
        train, val, test = partition_drug(df, drug)
        timestamp = log("OK", timestamp)

    else:
        print(f"invalid type: {args.partition}")
        exit(1)

    return train, val, test

def partition_random(df):
    train, val_test = train_test_split(df, test_size=0.2)
    val, test = train_test_split(val_test, test_size=0.5)
    del val_test
    return train, val, test

def partition_drug(df, drug):
    log("partition_drug...")
    specific_value = drug
    test = df[df.iloc[:, 1] == specific_value]  # Assuming column 2 is index 1
    mask = df.iloc[:, 1] != specific_value  # Assuming column 3, improve_chem_id,  is index 2
    filtered_df = df[mask]
    train, val = train_test_split(filtered_df, test_size=0.2)
    log_size("train", len(train))
    log_size("val",   len(val))
    log_size("test",  len(test))
    return train, val, test


def log_size(token, n):
    # Bring in comma-separated numbers:
    import locale
    locale.setlocale(locale.LC_ALL, "")
    token = token + ":"
    log(f"size: {token:6} {n:8n}")


def partition_cell(df, cell):
    specific_value = cell
    test = df[df.iloc[:, 13] == specific_value]
    mask = df.iloc[:, 13] != specific_value  # Assuming column 14, improve_sample_id,  is index 13
    filtered_df = df[mask]
    train, val = train_test_split(filtered_df, test_size=0.2)
    return train, val, test

def select_drug(index, df):
    '''
    Select the drug at given index==rank in an np.ndarray of unique drugs.
    This assumes np unique behaves the same all the time. TODO: sort
    If rank is greater than max index, NO: rank % max index is applied.
       crash!
    return: string: e.g., "Drug_123"
    '''

    unique_drugs = None
    unique_drugs = df['improve_chem_id'].unique() # change to df.iloc
    total = len(unique_drugs)
    if index is None:
        raise ValueError("provide an index for select by drug!")
    if index < 0:      raise IndexError("index too small!")
    if index >= total: raise IndexError("index too big!")

    log(f'selecting drug {index}/{total} ...')
    drug = unique_drugs[index]

    return drug

def select_cell(rank, df):
    ''' Select the cell at index==rank in an np.ndarray of unique cells.'''
    ''' This assumes np unique behaves the same all the time. TODO: sort'''
    idx = rank - 1
    unique_cells = None
    unique_cells = df['improve_sample_id'].unique() # change to df.iloc

    print(f'selecting cell at position {idx} for rank {rank} in unique_cells {len(unique_cells)}')
    if idx >= len(unique_cells):
        idx = (idx % len(unique_cells))
        print("idx greater than len uniq cells, resetting to {}".format(idx))
    print(f'selecting cell at position {idx} for rank {rank} in unique_cells {len(unique_cells)}')

    return unique_cells[idx]


def write(args, train, val, test):
    if "@@@" not in args.out:
        raise ValueError("out pattern does not contain @@@")
    outfile = out_file_name(args.out, "train")
    utils.write_df_pq(outfile, train)
    outfile = out_file_name(args.out, "val")
    utils.write_df_pq(outfile, val)
    outfile = out_file_name(args.out, "test")
    utils.write_df_pq(outfile, test)


def out_file_name(pattern, token):
    return pattern.replace("@@@", token) + ".parquet"


def log(txt, last_time=None):
    timestamp = utils.log("partition_uno: " + txt, last_time)
    return timestamp


def abort(txt, last_time=None):
    log("ABORT: " + txt, last_time)
    exit(1)


# # df = read_data(infile)
# _t = log(f'DONE_READING: {args.infile}')

# x_train_0 = train.iloc[:,14:972].copy()
# x_train_1 = train.iloc[:, 972:].copy()

# x_val_0 = val.iloc[:,14:972].copy()
# x_val_1 = val.iloc[:, 972:].copy()

# x_test_0 = test.iloc[:,14:972].copy()
# x_test_1 = test.iloc[:, 972:].copy()

# y_train = train[['auc','improve_sample_id','improve_chem_id']].copy()
# y_val = val[['auc','improve_sample_id','improve_chem_id']].copy()
# y_test = test[['auc','improve_sample_id','improve_chem_id']].copy()

# y_train.rename(columns={'auc':'AUC','improve_sample_id':'Sample','improve_chem_id':'Drug1'}, inplace=True)
# y_val.rename(columns={'auc':'AUC','improve_sample_id':'Sample','improve_chem_id':'Drug1'}, inplace=True)
# y_test.rename(columns={'auc':'AUC','improve_sample_id':'Sample','improve_chem_id':'Drug1'}, inplace=True)

# _t = log (f'DONE_CREATING_HDF_KEY_VALUES', _t)

# with pd.HDFStore(outfile, 'w') as f:
#       f['x_train_0'] = x_train_0
#       f['x_val_0'] = x_val_0
#       f['x_test_0'] = x_test_0

#       f['x_train_1'] = x_train_1
#       f['x_val_1'] = x_val_1
#       f['x_test_1'] = x_test_1

#       f['y_train'] = y_train
#       f['y_val'] = y_val
#       f['y_test'] = y_test

#       f['model'] = pd.DataFrame()

#       cl_width = x_train_0.shape[1]
#       dd_width = x_train_1.shape[1]

#       f.get_storer("model").attrs.input_features = OrderedDict(
#         [("cell.rnaseq", "cell.rnaseq"), ("drug1.descriptors", "drug.descriptors")] )
#       f.get_storer("model").attrs.feature_shapes = OrderedDict(
#         [("cell.rnaseq", (cl_width,)), ("drug.descriptors", (dd_width,))] )

# # f.close()
# log("DONE_SAVING_H5", _t)
# log("RUNTIME TOTAL", _s)

if __name__ == "__main__": main()
