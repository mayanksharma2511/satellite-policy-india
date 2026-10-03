"""Build the data files for the village progress tracker website (docs/).

For every village in the six states it combines:
  - where the village stood at the last survey (SECC 2011-12): poverty rate, consumption
  - how its night-time brightness has changed since: 2012-14 average vs latest three years
  - whether and when it got a PMGSY road (records end in 2015)

Within each district, villages are ranked on 2012 poverty and on brightening since.
"Left behind" = among the poorest 30% of the district in 2012 AND among the 30% darkest
at night today (average of the latest three years) (villages of 100+ people only). These are the villages to check first, not a verdict: night
lights pick up electrification and large changes, not small changes in income.

Also writes the numbers behind the "How far to trust this" page.

Run: python src/build_tracker.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

VILLAGES = Path("data/processed/villages.csv.gz")
LIGHTS = Path("data/processed/lights_by_year.csv")
OUT = Path("docs/data")
POOR, DARK = 0.70, 0.30  # poorest 30% in 2012, darkest 30% today
MIN_POP = 100  # survey rates for tiny hamlets are too noisy to flag


def main():
    v = pd.read_csv(VILLAGES, dtype={"shrid2": str})
    lights = pd.read_csv(LIGHTS, dtype={"shrid2": str})
    years = [int(c[-4:]) for c in lights.columns if c.startswith("lights_")]
    d = v.merge(lights, on="shrid2", how="inner")
    d = d.dropna(subset=["secc_pov_rate_rural", "latitude", "longitude"]).copy()

    base = d[[f"lights_{y}" for y in years[:3]]].mean(axis=1)
    now = d[[f"lights_{y}" for y in years[-3:]]].mean(axis=1)
    d["base"], d["now"] = base, now
    d["growth"] = np.log1p(now.clip(lower=0)) - np.log1p(base.clip(lower=0))
    district = d["state_name"] + "|" + d["district_name"]
    d["poor_rank"] = d.groupby(district)["secc_pov_rate_rural"].rank(pct=True)
    d["growth_rank"] = d.groupby(district)["growth"].rank(pct=True)
    d["now_rank"] = d.groupby(district)["now"].rank(pct=True)
    big = d["pc11_pca_tot_p"].fillna(0) >= MIN_POP
    d["left_behind"] = big & (d["poor_rank"] >= POOR) & (d["now_rank"] <= DARK)
    d["poor_rising"] = big & (d["poor_rank"] >= POOR) & (d["now_rank"] >= 1 - DARK)
    d["road_year"] = pd.to_datetime(d["road_comp_date_new"], errors="coerce").dt.year

    (OUT / "d").mkdir(parents=True, exist_ok=True)
    index = []
    fields = ["name", "lat", "lon", "pop", "pov", "cons", "road", "base", "now", "poor_rank",
              "growth_rank", "flag", "lights"]
    for (state, dist), g in d.groupby(["state_name", "district_name"]):
        did = f"{state}_{dist}".lower().replace(" ", "-").replace("/", "-")
        rows = []
        for r in g.itertuples():
            flag = 2 if r.left_behind else (1 if r.poor_rising else 0)
            rows.append([
                str(r.village_name).title() if pd.notna(r.village_name) else "",
                round(r.latitude, 5), round(r.longitude, 5),
                int(r.pc11_pca_tot_p) if pd.notna(r.pc11_pca_tot_p) else None,
                round(r.secc_pov_rate_rural, 3),
                int(r.secc_cons_pc_rural) if pd.notna(r.secc_cons_pc_rural) else None,
                int(r.road_year) if pd.notna(r.road_year) else None,
                round(r.base, 3), round(r.now, 3),
                round(r.poor_rank, 3), round(r.growth_rank, 3), flag,
                [round(getattr(r, f"lights_{y}"), 2) for y in years],
            ])
        (OUT / "d" / f"{did}.json").write_text(json.dumps({"fields": fields, "years": years,
                                                           "villages": rows}, separators=(",", ":")))
        index.append({"id": did, "state": state.title(), "district": str(dist).title(),
                      "villages": len(g), "left_behind": int(g["left_behind"].sum()),
                      "lat": round(g["latitude"].median(), 4), "lon": round(g["longitude"].median(), 4)})
    (OUT / "districts.json").write_text(json.dumps(index, separators=(",", ":")))

    # numbers for the "How far to trust this" page
    rd = pd.read_csv("results/rd_roads.csv")
    main_bw = rd[rd["bandwidth"] == 84].set_index("outcome")
    se = lambda o: float((main_bw.loc[o, "ci_high"] - main_bw.loc[o, "ci_low"]) / 3.92)
    event = pd.read_csv("results/lights_after_road.csv")
    trust = {
        "rd": {"survey": {"est": float(main_bw.loc["survey", "estimate"]), "se": se("survey")},
               "satellite": {"est": float(main_bw.loc["satellite", "estimate"]), "se": se("satellite")},
               "first_stage": float(main_bw.loc["survey", "first_stage"]),
               "n": int(main_bw.loc["survey", "n"]),
               "cultiv": {"est": float(main_bw.loc["cultiv_share_12", "estimate"]), "se": se("cultiv_share_12")}},
        "shrink": 0.208,
        "lights_after_road": event.to_dict(orient="list"),
        "summary": {"villages": int(len(d)), "districts": len(index),
                    "left_behind": int(d["left_behind"].sum()), "years": [years[0], years[-1]]},
    }
    (OUT / "trust.json").write_text(json.dumps(trust, indent=1))
    print(f"{len(d):,} villages in {len(index)} districts; "
          f"{int(d['left_behind'].sum()):,} flagged as left behind. Wrote {OUT}/")


if __name__ == "__main__":
    main()
