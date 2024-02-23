import pandas as pd
import h5py
import sys
from sklearn.model_selection import train_test_split
import numpy as np

def read_data(path_to_all_source):
    df = pd.read_parquet(path_to_all_source)
    df_drug = df.filter(regex=r'^mordred\.', axis=1)
    df_cell_auc = df.loc[:, ~df.columns.str.startswith('mordred')]

    df_cell = df_cell_auc.loc[:, df_cell_auc.columns.str.isupper()]

    df_auc = df_cell_auc.loc[:, ~df_cell_auc.columns.str.isupper()]

    # df_auc.rename(columns={'improve_chem_id': 'Drug1'}, inplace=True)
    # df_auc.rename(columns={'source': 'Sample'}, inplace=True)
    # df_auc.rename(columns={'auc': 'AUC'}, inplace=True)
    # df_auc = df_auc.filter(like=['AUC'], axis=1)
    df_auc = df_auc.loc[:, ['auc']]
    df_auc.rename(columns={'auc': 'AUC'}, inplace=True)
    # df_auc

    return df_auc, df_cell, df_drug


def partition(path_to_all_source, random_state, test_size=0.2):
    """
    Split the dataframe into train and test sets.
    """
    df_auc, df_cell, df_drug = read_data(path_to_all_source)
    print("Done reading file and splitting into dataframes.")
   # Split df_cell into train, val, and test sets
    train_cell, test_val_cell = train_test_split(df_cell, test_size=0.2, random_state=42)
    val_cell, test_cell = train_test_split(test_val_cell, test_size=0.5, random_state=42)


   # Split df_auc into train, val, and test sets
    train_auc, test_val_auc = train_test_split(df_auc, test_size=0.2, random_state=42)
    val_auc, test_auc = train_test_split(test_val_auc, test_size=0.5, random_state=42)


   # Split df_drug into train, val, and test sets
    train_drug, test_val_drug = train_test_split(df_drug, test_size=0.2, random_state=42)
    val_drug, test_drug = train_test_split(test_val_drug, test_size=0.5, random_state=42)
    
    
    print("Now saving to HDF5 file", train_cell.shape, val_cell.shape, test_cell.shape, train_drug.shape, val_drug.shape, test_drug.shape, train_auc.shape, val_auc.shape, test_auc.shape)
   # Saving to HDF5 file
    with pd.HDFStore('data.h5', 'w') as f:
      f['x_train_0'] = train_cell
      f['x_val_0'] = val_cell
      f['x_test_0'] = test_cell
      
      f['x_train_1'] = train_drug
      f['x_val_1'] = val_drug
      f['x_test_1'] = test_drug
    
      f['y_train'] = train_auc
      f['y_val'] = val_auc
      f['y_test'] = test_auc

      # Create an empty DataFrame with zero rows and num_columns columns
      empty_df = pd.DataFrame()
      f['model'] = empty_df
    
    print("Done saving to HDF5 file")

if __name__ == "__main__":
   print("Starting the process")

   if len(sys.argv) > 1:
    path_to_all_source = sys.argv[1]
   else:
    path_to_all_source = '/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE/test/data_geneexp_mordred_all_source.parquet'

   partition(path_to_all_source, random_state=42, test_size=0.2)
   
  #  print keys in the o/p file:
   with pd.HDFStore('data.h5', 'r') as f:
    print(f.keys())
    # print content of key called model in the o/p file:
    # print(f['x_test_0'])
    print(f['model'])
   
  #  path_to_all_source = '/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE/test/data_geneexp_mordred_all_source.parquet'
   
  # TODO: create 'partition_cell_drug' by cell and drug based on a values specified (gene or sp composition spefified) in a file
  # TODO: create partition_procs - use rank and total number of procs to create different h5 files for use with --use-exported function in UNO Benchmark Pilot1 of CANDLE/Benchmarks
  # Combine the above two to create partition_cell_drug_procs - will be very useful for large scale runs