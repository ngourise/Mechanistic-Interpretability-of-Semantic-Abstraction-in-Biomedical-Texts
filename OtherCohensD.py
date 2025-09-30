import pandas as pd
import numpy as np
from google.colab import files

#Change any instance of "dmis-lab/biobert-base-cased-v1" to "allenai/scibert_scivocab_uncased" or "hossboll/clinical-t5" depending on which model this is being run for.
#Code below is set up for bioBERT.

uploaded = files.upload()
dfs = [pd.read_csv(f) for f in uploaded.keys()]
df = pd.concat(dfs, ignore_index=True)
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                          (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std
biobert_df = df[df["model"] == "dmis-lab/biobert-base-cased-v1"]
results = []
for component, group in biobert_df.groupby("component"):
    suff_scores = group.loc[group["mode"] == "sufficiency", "impact_score"]
    rand_scores = group.loc[group["mode"] == "random", "impact_score"]

    if len(suff_scores) > 1 and len(rand_scores) > 1:
        d = cohens_d(suff_scores, rand_scores)
        results.append({"component": component, "cohens_d": d})

biobert_results = pd.DataFrame(results)
print("Cohen's D for BioBERT (dmis-lab/biobert-base-cased-v1):")
display(biobert_results)


