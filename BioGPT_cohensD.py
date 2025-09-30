import pandas as pd
import numpy as np
from google.colab import files
uploaded = files.upload()
dfs = [pd.read_csv(f) for f in uploaded.keys()]
df = pd.concat(dfs, ignore_index=True)
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std
results = []
for component, group in df.groupby("patch_type"):
    suff_scores = group.loc[group["strategy"] == "sufficiency", "delta_ppl"]
    rand_scores = group.loc[group["strategy"] == "random", "delta_ppl"]
    if len(suff_scores) > 1 and len(rand_scores) > 1:
        d = cohens_d(suff_scores, rand_scores)
        results.append({"component": component, "cohens_d": d})
results_df = pd.DataFrame(results)
results_df
