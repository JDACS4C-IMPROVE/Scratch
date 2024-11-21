
import pandas as pd

d = "exp_result/"

df_rsp = pd.read_parquet(d + "rsp_train_data.parquet")
print("rsp: %i" % len(df_rsp))
# print(str(df_rsp))

L = [ i for i in range(0, 3000) if i % 2 == 1 ]

# print(L)

df_rsp.drop(labels=L, inplace=True)
print(str(len(df_rsp)))

df_rsp.to_parquet(path="cut.parquet")
