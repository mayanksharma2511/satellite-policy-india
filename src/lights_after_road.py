"""Do villages get brighter at night after they get a road? A check on the tracker's signal.

If night lights measure real change, villages that got a PMGSY road should brighten
afterwards compared with similar villages that were still waiting for theirs.

  treated   villages whose road was completed in 2013 or 2014
  control   villages sanctioned a road in 2011-2013 but not completed by 2015
            (they got it later, so they are similar villages a few years behind)

Event study: log(1 + brightness) by year relative to the road, with village fixed
effects and district-by-year fixed effects (so only villages in the same district in the
same year are compared). The year before the road is the reference. Standard errors are
clustered by district. Flat estimates before the road are the check that the two groups
were on the same path beforehand.

Run: python src/lights_after_road.py   ->  results/lights_after_road.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd


def demean(frame, groups, iterations=30):
    """Remove two sets of fixed effects by alternating projections."""
    for _ in range(iterations):
        for g in groups:
            frame = frame - frame.groupby(g).transform("mean")
    return frame


def main():
    v = pd.read_csv("data/processed/villages.csv.gz", dtype={"shrid2": str})
    lights = pd.read_csv("data/processed/lights_by_year.csv", dtype={"shrid2": str})
    d = v.merge(lights, on="shrid2")
    d["road_year"] = pd.to_datetime(d["road_comp_date_new"], errors="coerce").dt.year
    treated = d["road_year"].isin([2013, 2014])
    control = d["road_year"].isna() & d["road_sanc_year_new"].between(2011, 2013)
    s = d[treated | control].copy()
    s["treated"] = treated[treated | control]
    print(f"Treated villages: {s['treated'].sum():,}; comparison villages: {(~s['treated']).sum():,}")

    cols = [c for c in lights.columns if c.startswith("lights_")]
    long = s.melt(id_vars=["shrid2", "state_name", "district_name", "road_year", "treated"],
                  value_vars=cols, var_name="year", value_name="lights")
    long["year"] = long["year"].str[-4:].astype(int)
    long["y"] = np.log1p(long["lights"].clip(lower=0))
    district = (long["state_name"] + "|" + long["district_name"]).to_numpy()
    long["event"] = np.where(long["treated"], long["year"] - long["road_year"], np.nan)

    events = [e for e in range(-2, 13) if e != -1]
    X = pd.DataFrame({e: (long["event"] == e).astype(float) for e in events})
    frame = demean(pd.concat([long[["y"]], X], axis=1),
                   [long["shrid2"].to_numpy(), district + long["year"].astype(str).to_numpy()])
    y, X = frame["y"].to_numpy(), frame[events].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    scores = pd.DataFrame(X * resid[:, None]).groupby(district).sum().to_numpy()
    se = np.sqrt(np.diag(bread @ (scores.T @ scores) @ bread))

    out = pd.DataFrame({"event_time": events, "estimate": beta, "se": se})
    out = pd.concat([out, pd.DataFrame({"event_time": [-1], "estimate": [0.0], "se": [0.0]})])
    out = out.sort_values("event_time")
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/lights_after_road.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
