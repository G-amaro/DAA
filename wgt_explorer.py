import pandas as pd
tr = pd.read_csv("datasets/dataset_train.csv", dtype=str)
te = pd.read_csv("datasets/dataset_test_X.csv", dtype=str)
w = pd.concat([tr["WGT"], te["WGT"]])
print("valores únicos:", w.nunique(), "de", len(w))
print(w.value_counts().value_counts().sort_index().head(15))   # tamanho dos grupos
tr["AGE_RANGE"] = tr["AGE_RANGE"].astype(int)
g = tr.groupby("WGT")["AGE_RANGE"].agg(["size", "min", "max"])
g = g[g["size"] > 1]
print("grupos com >1 pessoa no treino:", len(g))
print((g["max"] - g["min"]).describe())                         # amplitude de classes por grupo