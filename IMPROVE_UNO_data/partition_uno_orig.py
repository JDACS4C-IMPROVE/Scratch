
"""
PARTITION UNO
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
import sys

import argparse
parser = argparse.ArgumentParser(description="description")
parser.add_argument("--partition", type=str, default="random",
                    help="partition type (random, by_cell, by_drug)"
                    )
parser.add_argument("--pals_rank", type=int, default=0,
                    help="rank to be used as index to cell or drug"
                    )
parser.add_argument("--infile", type=str, default="merged.landmark.h5",
                    help="name of input file"
                    )
parser.add_argument("--outfile", type=str, default="outfile",
                    help="name of output file"
                    )
args = parser.parse_args()
print(f'using partition: {args.partition}')

partition=args.partition
infile=args.infile
outfile=args.outfile
rank=args.pals_rank

import datetime
import time
def _print(txt="HERE", last_time=time.time()):
    this_time = time.time()
    print("{}\t{}\t{}".format(datetime.datetime.now(),
                              this_time-last_time,
                              txt)
         )
    return time.time()

_s = _print(f'DONE_LOADING_LIBS')

def read_data(f):
    store = pd.HDFStore(f, "r")
    df = store['df']
    store.close()
    return df

def partition_random(df):
    train, val_test = train_test_split(df, test_size=0.2)
    val, test = train_test_split(val_test, test_size=0.5)
    del val_test
    return train, val, test

def partition_drug(df, drug):
    specific_value = drug
    test = df[df.iloc[:, 1] == specific_value]  # Assuming column 2 is index 1
    mask = df.iloc[:, 1] != specific_value  # Assuming column 3, improve_chem_id,  is index 2
    filtered_df = df[mask]
    train, val = train_test_split(filtered_df, test_size=0.2)
    return train, val, test

def partition_cell(df, cell):
    specific_value = cell
    test = df[df.iloc[:, 13] == specific_value]
    mask = df.iloc[:, 13] != specific_value  # Assuming column 14, improve_sample_id,  is index 13
    filtered_df = df[mask]
    train, val = train_test_split(filtered_df, test_size=0.2)
    return train, val, test

def select_drug(rank, df):
    ''' Select the drug at index==rank in an np.ndarray of unique drugs.'''
    ''' This assumes np unique behaves the same all the time. TODO: sort'''
    ''' If rank is greater than max index, rank % max index is applied.'''
    idx = rank - 1
    assert idx >= 0

    unique_drugs = None
    unique_drugs = df['improve_chem_id'].unique() # change to df.iloc

    print(f'selecting drug at position {idx} for rank {rank} in unique_drugs {len(unique_drugs)}')
    if idx >= len(unique_drugs):
        idx = (idx % len(unique_drugs))
        print("idx greater than len uniq drugs, resetting to {}".format(idx))
    print(f'selecting drug at position {idx} for rank {rank} in unique_drugs {len(unique_drugs)}')
    return unique_drugs[idx]

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

df = read_data(infile)
_t = _print(f'DONE_READING_{infile}', _s)


if partition == "random":
    train, val, test = partition_random(df)
    _t = _print(f'DONE_PARTITIONING_DATA_{infile}', _t)

elif partition == "by_cell":
    cell = select_cell(rank, df=df)
    _t = _print("DONE_SELECTING_CELL", _t)
    print(cell)
    train, val, test = partition_cell(df, cell)
    _t = _print(f'DONE_PARTITIONING_DATA_{infile}', _t)

elif partition == "by_drug":
    drug = select_drug(rank, df=df)
    _t = _print("DONE_SELECTING_DRUG", _t)
    print(drug)
    train, val, test = partition_drug(df, drug)
    _t = _print(f'DONE_PARTITIONING_DATA_{infile}', _t)

else:
    print(f"invalid type{partition}")
    sys.exit()

x_train_0 = train.iloc[:,14:972].copy()
x_train_1 = train.iloc[:, 972:].copy()

x_val_0 = val.iloc[:,14:972].copy()
x_val_1 = val.iloc[:, 972:].copy()

x_test_0 = test.iloc[:,14:972].copy()
x_test_1 = test.iloc[:, 972:].copy()

y_train = train[['auc','improve_sample_id','improve_chem_id']].copy()
y_val = val[['auc','improve_sample_id','improve_chem_id']].copy()
y_test = test[['auc','improve_sample_id','improve_chem_id']].copy()

y_train.rename(columns={'auc':'AUC','improve_sample_id':'Sample','improve_chem_id':'Drug1'}, inplace=True)
y_val.rename(columns={'auc':'AUC','improve_sample_id':'Sample','improve_chem_id':'Drug1'}, inplace=True)
y_test.rename(columns={'auc':'AUC','improve_sample_id':'Sample','improve_chem_id':'Drug1'}, inplace=True)

_t = _print (f'DONE_CREATING_HDF_KEY_VALUES', _t)

with pd.HDFStore(outfile, 'w') as f:
      f['x_train_0'] = x_train_0
      f['x_val_0'] = x_val_0
      f['x_test_0'] = x_test_0

      f['x_train_1'] = x_train_1
      f['x_val_1'] = x_val_1
      f['x_test_1'] = x_test_1

      f['y_train'] = y_train
      f['y_val'] = y_val
      f['y_test'] = y_test

      f['model'] = pd.DataFrame()

      cl_width = x_train_0.shape[1]
      dd_width = x_train_1.shape[1]

      f.get_storer("model").attrs.input_features = OrderedDict(
        [("cell.rnaseq", "cell.rnaseq"), ("drug1.descriptors", "drug.descriptors")] )
      f.get_storer("model").attrs.feature_shapes = OrderedDict(
        [("cell.rnaseq", (cl_width,)), ("drug.descriptors", (dd_width,))] )

f.close()
_print("DONE_SAVING_H5", _t)
_print("RUNTIME TOTAL", _s)
