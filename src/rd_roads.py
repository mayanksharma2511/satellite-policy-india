"""Re-run Asher & Novosad's rural-roads regression discontinuity with satellite estimates.

Villages just above a PMGSY population cut-off were much more likely to get a new road
than villages just below it, but are otherwise similar. Comparing the two groups gives
the effect of a road (a "fuzzy" regression discontinuity: the cut-off raises the chance
of a road rather than guaranteeing it, so the road effect = jump in outcome / jump in roads).

We estimate the effect of a road on village consumption twice, on the same villages:
  survey     log consumption per person from the SECC (what the paper used)
  satellite  the CNN's cross-fitted estimate of the same thing
and ask: would the satellite version have reached the same conclusion?

Also reported:
  - first stage (how much the cut-off raises the chance of a road)
  - share of workers in agriculture, 2011 (the paper's clearest effect) as a sanity check
  - minimum detectable effect: the smallest road effect this design could reliably detect
  - road-visibility check: does the satellite model's *error* jump at the cut-off?
    If it does, the model may be "seeing" the new road itself rather than wealth.
  - robustness to the bandwidth (60, 84, 120 people)

Specification (following the paper): local linear regression on each side of the
cut-off, triangular kernel, state-by-cut-off fixed effects, baseline 2001 controls,
heteroskedasticity-robust standard errors.

Run: python src/rd_roads.py --predictions outputs/predictions_crossfit.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

VILLAGES = Path("data/processed/villages.csv.gz")
RESULTS = Path("results")
MAIN_BANDWIDTH = 84
CONTROLS = ["lit_rate_01", "sc_share_01", "st_share_01", "ag_share_01",
            "power_01", "dist_town_01", "log_pop_01"]


def prepare(villages, predictions):
    v = villages.merge(predictions, on="shrid2", how="left")
    p = v["pc01_pca_tot_p"]
    work = v["pc01_pca_tot_work_p"].replace(0, np.nan)
    v["lit_rate_01"] = v["pc01_pca_p_lit"] / p
    v["sc_share_01"] = v["pc01_pca_p_sc"] / p
    v["st_share_01"] = v["pc01_pca_p_st"] / p
    v["ag_share_01"] = (v["pc01_pca_main_cl_p"] + v["pc01_pca_main_al_p"]) / work
    v["power_01"] = v["pc01_vd_power_supl"].fillna(0).clip(0, 1)
    v["dist_town_01"] = v["pc01_vd_dist_town"]
    v["log_pop_01"] = np.log(p)
    v["ag_share_11"] = ((v["pc11_pca_main_cl_p"] + v["pc11_pca_main_al_p"])
                        / v["pc11_pca_tot_work_p"].replace(0, np.nan))
    v["survey"] = np.log(v["secc_cons_pc_rural"])
    v["satellite"] = v["pred_cnn"]
    v["satellite_error"] = v["satellite"] - v["survey"]
    if "pred_ridge" in v:
        v["ridge"] = v["pred_ridge"]
    v["road"] = v["new_road_by_2012"]
    v["group"] = v["state_name"] + "_" + v["threshold"].astype(int).astype(str)
    v["r_above"] = v["running"] * v["above"]
    return v


def sample(v, bandwidth):
    s = v[(v["no_paved_road_2001"] == 1) & (v["running"].abs() <= bandwidth)].copy()
    s["w"] = 1 - s["running"].abs() / bandwidth  # triangular kernel: closest villages count most
    return s.dropna(subset=CONTROLS)


def rhs():
    return " + ".join(["running", "r_above", "C(group)", *CONTROLS])


def first_stage(s):
    fit = smf.wls(f"road ~ above + {rhs()}", data=s, weights=s["w"]).fit(cov_type="HC1")
    return fit.params["above"], fit.bse["above"]


def jump(s, outcome):
    """Reduced form: how much the outcome jumps at the cut-off."""
    d = s.dropna(subset=[outcome])
    fit = smf.wls(f"{outcome} ~ above + {rhs()}", data=d, weights=d["w"]).fit(cov_type="HC1")
    return fit.params["above"], fit.bse["above"], int(fit.nobs)


def road_effect(s, outcome):
    """Fuzzy RD: effect of getting a road, instrumenting road with being above the cut-off."""
    d = s.dropna(subset=[outcome])
    fit = IV2SLS.from_formula(f"{outcome} ~ 1 + {rhs()} + [road ~ above]",
                              data=d, weights=d["w"]).fit(cov_type="robust")
    est, se = fit.params["road"], fit.std_errors["road"]
    return {"estimate": est, "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
            "p_value": fit.pvalues["road"], "mde_80": 2.8 * se, "n": int(fit.nobs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/predictions_crossfit.csv")
    parser.add_argument("--villages", default=str(VILLAGES))
    args = parser.parse_args()

    villages = pd.read_csv(args.villages, dtype={"shrid2": str})
    predictions = pd.read_csv(args.predictions, dtype={"shrid2": str})
    v = prepare(villages, predictions)
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for bw in (60, MAIN_BANDWIDTH, 120):
        s = sample(v, bw)
        # compare survey and satellite on exactly the same villages
        same = s.dropna(subset=["survey", "satellite"])
        fs, fs_se = first_stage(same)
        outcomes = ["survey", "satellite", "ag_share_11"] + (["ridge"] if "ridge" in s else [])
        for outcome in outcomes:
            rows.append({"bandwidth": bw, "outcome": outcome, "first_stage": fs,
                         "first_stage_se": fs_se, **road_effect(same, outcome)})
        err, err_se, n = jump(same, "satellite_error")
        rows.append({"bandwidth": bw, "outcome": "satellite_error_jump", "estimate": err,
                     "se": err_se, "ci_low": err - 1.96 * err_se, "ci_high": err + 1.96 * err_se,
                     "n": n})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "rd_roads.csv", index=False)

    main = out[out["bandwidth"] == MAIN_BANDWIDTH].set_index("outcome")
    print(f"\nMain bandwidth ({MAIN_BANDWIDTH} people), villages with both measures: "
          f"{int(main.loc['survey', 'n']):,}")
    print(f"First stage: being above the cut-off raises the chance of a road by "
          f"{main.loc['survey', 'first_stage']:.3f} (se {main.loc['survey', 'first_stage_se']:.3f})")
    print("\nEffect of a new road (log consumption per person; 0.10 is about +10%):")
    for name in ["survey", "satellite", "ridge", "ag_share_11"]:
        if name in main.index:
            r = main.loc[name]
            print(f"  {name:<12} {r['estimate']:+.3f}  95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                  f"  smallest detectable effect {r['mde_80']:.3f}")
    e = main.loc["satellite_error_jump"]
    print(f"\nRoad-visibility check: satellite error jumps by {e['estimate']:+.3f} at the cut-off "
          f"(95% CI [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]); a CI that excludes 0 is a warning.")
    print(f"\nSaved {RESULTS / 'rd_roads.csv'} (all bandwidths)")


if __name__ == "__main__":
    main()
