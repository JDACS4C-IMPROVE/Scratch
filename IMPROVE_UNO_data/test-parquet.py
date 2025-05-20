import pandas as pd

d = "exp_result/"

df_ge = pd.read_parquet(d + "ge_test_data.parquet")
print("ge: %i" % len(df_ge))

print(str(df_ge))

df_md = pd.read_parquet(d + "md_test_data.parquet")
print("md: %i" % len(df_md))
print(str(df_md))

df_rsp = pd.read_parquet(d + "rsp_test_data.parquet")
print("rsp: %i" % len(df_rsp))
print(str(df_rsp))

