# Project notes (status and next steps)

## Status (3 Oct 2026)
- Village table: 217,305 villages, six PMGSY threshold states; roads sample 17,046 (no paved road in 2001, within 84 of a cut-off).
- Tiles: 26,040 Landsat 7 tiles (Oct 2010 - Mar 2012 dry-season median, 96 x 96 px, 6 bands).
- CNN (ResNet-18, 8 epochs, Kaggle T4), label log SECC consumption per person:
  - cross-fit by district: R2 0.199 (ridge baseline 0.140); within-state Spearman 0.40 (0.34)
  - unseen state: R2 0.088 (0.072); within-state Spearman 0.34 (0.30)
  - accuracy loss in new states is mostly a level shift, not a loss of ranking
- Roads RD (district x cut-off FE, triangular kernel, bandwidth 84, n = 16,024):
  - first stage 0.162 (paper 0.217)
  - SECC cultivation share: -0.141, 95% CI [-0.291, +0.010]; paper -0.092 -> consistent
  - consumption, survey: -0.016 [-0.115, +0.082]; satellite: -0.016 [-0.057, +0.025]
  - road-visibility check: no jump in satellite error at the cut-off
  - attenuation: within-state slope of satellite estimate on survey = 0.21, so a true
    effect shows up at about a fifth of its size; the satellite design could only detect a
    true consumption effect of roughly 0.059 / 0.21 = 0.28 (28%), vs 0.14 for the survey.
    The satellite "agreement" with the null is therefore weak evidence.

## Next steps (decide on validation data only, report every run)
1. Model improvements, each run once: add a night-lights band (VIIRS 2012 / DMSP 2011);
   train 20 epochs; consider a housing/assets label (Census 2011 houselisting).
2. Re-run evaluate_predictions.py and rd_roads.py; report slope and detectable effect.
3. Public map, write-up, README.
