"""How well do the satellite estimates track survey consumption?

Reports, for the cross-fitted and leave-one-state-out predictions:
  R2           overall fit (penalises a model that is shifted up/down in a new state)
  R2_demeaned  fit after removing each state's average (ignores that shift)
  spearman     rank correlation within each state: does it order villages correctly?

Run: python src/evaluate_predictions.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

villages = pd.read_csv("data/processed/villages.csv.gz", dtype={"shrid2": str},
                       usecols=["shrid2", "state_name", "secc_cons_pc_rural"])
rows = []
for name in ("crossfit", "loso"):
    p = pd.read_csv(f"outputs/predictions_{name}.csv", dtype={"shrid2": str})
    d = p.merge(villages, on="shrid2")
    d["y"] = np.log(d["secc_cons_pc_rural"])
    d = d.dropna(subset=["y", "pred_cnn"])
    for model in ("pred_cnn", "pred_ridge"):
        for state, g in [("all", d), *d.groupby("state_name")]:
            y, f = g["y"], g[model]
            dm = (g.groupby("state_name")["y"].transform("mean"), g.groupby("state_name")[model].transform("mean"))
            rows.append({"test": name, "model": model.replace("pred_", ""), "state": state, "n": len(g),
                         "R2": r2_score(y, f), "R2_demeaned": r2_score(y - dm[0], f - dm[1]),
                         "spearman": g.groupby("state_name").apply(
                             lambda s: s["y"].corr(s[model], method="spearman"), include_groups=False).mean()})
out = pd.DataFrame(rows)
out.to_csv("outputs/prediction_quality.csv", index=False)
pd.set_option("display.width", 140)
print(out.pivot_table(index=["test", "state"], columns="model", values=["R2", "R2_demeaned", "spearman"]).round(3))
