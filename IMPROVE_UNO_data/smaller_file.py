import pandas as pd

path_to_all_source = '/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE/test/data_geneexp_mordred_all_source.parquet'
path_to_smaller_file = '.smaller_file.parquet'

# Read the original file
df = pd.read_parquet(path_to_all_source)

# Create a smaller DataFrame with 100 entries
smaller_df = df.head(100)

# Write the smaller DataFrame to a new file
smaller_df.to_parquet(path_to_smaller_file)
