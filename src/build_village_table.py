"""Build one table of villages for the six states in Asher & Novosad's roads study.

Each row is a SHRUG village (shrid). It joins:
  - location (state, district, latitude/longitude)       -> for cutting satellite images
  - 2001 census population and workforce                  -> the running variable
  - 2001 village directory (paved road, electricity)      -> who was eligible for a new road
  - 2011 census population and workforce                  -> outcomes
  - SECC 2012 consumption and poverty rate                 -> the wealth label the CNN learns
  - PMGSY: when a new road was sanctioned / completed     -> the "treatment"

Output: data/processed/villages.csv.gz

Run from the repository root:  python src/build_village_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/shrug")
OUT = Path("data/processed/villages.csv.gz")

# States that followed PMGSY's population thresholds (Asher & Novosad, Table 1).
# Chhattisgarh and Madhya Pradesh used both the 500 and 1,000 cut-offs; the others only 500.
THRESHOLDS = {
    "chhattisgarh": [500, 1000],
    "gujarat": [500],
    "madhya pradesh": [500, 1000],
    "maharashtra": [500],
    "odisha": [500],
    "rajasthan": [500],
}
BANDWIDTH = 84  # people either side of the cut-off, the paper's optimal bandwidth

COLUMNS = {
    "shrug-shrid-keys-csv/shrid2_spatial_stats.csv": [
        "latitude", "longitude", "area_laea", "high_quality",
    ],
    "shrug-pca01-csv/pc01_pca_clean_shrid.csv": [
        "pc01_pca_tot_p", "pc01_pca_no_hh", "pc01_pca_p_lit", "pc01_pca_p_sc", "pc01_pca_p_st",
        "pc01_pca_tot_work_p", "pc01_pca_main_cl_p", "pc01_pca_main_al_p",
    ],
    "shrug-vd01-csv/pc01_vd_clean_shrid.csv": [
        "pc01_vd_app_pr", "pc01_vd_power_supl", "pc01_vd_dist_town",
    ],
    "shrug-pca11-csv/pc11_pca_clean_shrid.csv": [
        "pc11_pca_tot_p", "pc11_pca_no_hh", "pc11_pca_p_lit",
        "pc11_pca_tot_work_p", "pc11_pca_main_cl_p", "pc11_pca_main_al_p",
    ],
    "shrug-secc-cons-rural-csv/secc_cons_rural_shrid.csv": [
        "secc_cons_pc_rural", "secc_pov_rate_rural",
    ],
    "shrug-pmgsy-csv/pmgsy_2015_shrid.csv": [
        "road_sanc_year_new", "road_award_date_new", "road_comp_date_new",
    ],
}


def read_for(path, columns, keep):
    """Read only the needed columns, in chunks, keeping only villages in `keep`."""
    parts = []
    for chunk in pd.read_csv(RAW / path, usecols=["shrid2", *columns], chunksize=200_000,
                             dtype={"shrid2": str}, low_memory=False):
        parts.append(chunk[chunk["shrid2"].isin(keep)])
    return pd.concat(parts).drop_duplicates("shrid2").set_index("shrid2")


def nearest_threshold(pop, state):
    cutoffs = np.array(THRESHOLDS[state])
    return cutoffs[np.argmin(np.abs(cutoffs - pop))]


def main():
    names = pd.read_csv(RAW / "shrug-shrid-keys-csv/shrid_loc_names.csv",
                        usecols=["shrid2", "state_name", "district_name", "village_name"],
                        dtype=str)
    names = names[names["state_name"].isin(THRESHOLDS)].drop_duplicates("shrid2")
    keep = set(names["shrid2"])
    table = names.set_index("shrid2")
    print(f"Shrids in the six states: {len(table):,}")

    for path, cols in COLUMNS.items():
        part = read_for(path, cols, keep)
        table = table.join(part, how="left")
        print(f"  joined {path.split('/')[-1]:<32} {part.shape[0]:>7,} matched")

    # Rural villages only: SECC rural consumption exists, and there was a 2001 population
    table = table[table["pc01_pca_tot_p"].notna()]

    # Running variable: 2001 population minus the nearest cut-off used in that state
    table["threshold"] = [nearest_threshold(p, s)
                          for p, s in zip(table["pc01_pca_tot_p"], table["state_name"])]
    table["running"] = table["pc01_pca_tot_p"] - table["threshold"]
    table["above"] = (table["running"] >= 0).astype(int)

    # Eligible for a *new* road: no paved road in 2001
    table["no_paved_road_2001"] = (table["pc01_vd_app_pr"] == 0).astype(int)

    # Treatment: a new PMGSY road completed before the 2011 census / 2012 SECC
    completed = pd.to_datetime(table["road_comp_date_new"], errors="coerce")
    table["new_road_by_2011"] = (completed < "2011-01-01").astype(int)
    table["new_road_by_2012"] = (completed < "2012-01-01").astype(int)

    table["in_rd_sample"] = ((table["no_paved_road_2001"] == 1)
                             & (table["running"].abs() <= BANDWIDTH)).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(OUT, index=False)

    # ---- sanity checks ----
    rd = table[table["in_rd_sample"] == 1]
    print(f"\nRural villages with 2001 population: {len(table):,}")
    print(f"  with coordinates:          {table['latitude'].notna().sum():,}")
    print(f"  with SECC consumption:     {table['secc_cons_pc_rural'].notna().sum():,}")
    print(f"  no paved road in 2001:     {table['no_paved_road_2001'].sum():,}")
    print(f"\nRoads-study sample (no paved road, within {BANDWIDTH} of a cut-off): {len(rd):,}")
    print("  (Asher & Novosad report 11,432 villages after their own matching)")
    print("\nVillages per state (all / roads-study sample):")
    print(pd.DataFrame({"all": table.groupby("state_name").size(),
                        "rd_sample": rd.groupby("state_name").size()}).to_string())
    print("\nFirst-stage check - share with a new road by 2011, just below vs just above the cut-off:")
    print(rd.groupby("above")["new_road_by_2011"].mean().rename({0: "below", 1: "above"}).round(3).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
