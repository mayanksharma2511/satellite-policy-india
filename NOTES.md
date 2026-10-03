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

## Run 2: 20 epochs (3 Oct 2026)
Same code and data, `--epochs 20`. Files kept in outputs/e20/ (8-epoch run in outputs/e8/).

| | 8 epochs | 20 epochs |
|---|---|---|
| Cross-fit R² | 0.199 | 0.205 |
| Cross-fit within-state Spearman | 0.40 | 0.41 |
| Unseen-state (LOSO) R² | 0.088 | 0.043 |
| Unseen-state Spearman | 0.34 | 0.34 |
| Shrink factor (RD sample) | 0.208 | 0.225 |
| RD satellite estimate | -0.016 | -0.015 |

Decision: keep 8 epochs as the main model. Longer training helps slightly in states the model
has seen but hurts in new ones (Chhattisgarh -0.05 -> -0.29), so it fits the training states
more closely without learning more about wealth. Training length is not the bottleneck;
the inputs are. Next: add night lights as an input.

## Run 3: night lights added (3 Oct 2026)
`src/export_nightlights.py` gets VIIRS 2012 (Apr-Dec mean) and DMSP 2011 brightness over each
tile's square. Fed to the CNN next to the image features, added to the ridge baseline, and
used alone as a third model. 8 epochs. Files in outputs/lights/.

| Model | Cross-fit R² | Within-state Spearman (cross-fit / unseen state) | Unseen-state R² | Shrink factor (RD sample) |
|---|---|---|---|---|
| CNN, images only (main) | 0.199 | 0.40 / 0.34 | 0.088 | 0.21 |
| CNN + night lights | 0.144 | 0.41 / 0.37 | 0.099 | 0.22 |
| Ridge + night lights | 0.187 | 0.40 / 0.37 | 0.114 | 0.16 |
| Night lights only | 0.101 | 0.34 / 0.33 | 0.068 | 0.09 |

- Night lights help most in states the model has not seen (ranking 0.34 -> 0.37).
- The CNN + lights R² drop comes from one cross-fit fold: in Rajasthan's desert districts
  (Barmer, Bikaner, Churu) it overestimates consumption by 0.6-0.7 log points, where the
  images-only model was unbiased. Ranking elsewhere is unaffected.
- RD with CNN + lights: -0.010, 95% CI [-0.053, +0.033]; no road-visibility jump.
- Ridge gives a "significant" negative road effect in every run (about -0.035, CI just
  excluding 0) that the survey does not show: a simple model can produce a false finding.

Decision: images-only CNN stays the main model; the others are reported as robustness.
Main finding: across every model tried, satellite estimates show only 9-22% of a true
difference in consumption, so the satellite RD could only detect large road effects.
Model tuning stops here. Next: write-up and map.

## Village progress tracker (3 Oct 2026)
- `src/export_lights_timeseries.py`: VIIRS (VCMCFG) yearly median brightness 2012-2025, 2 km square,
  all 217,305 villages -> data/processed/lights_by_year.csv. 2012 is Apr-Dec only; 2016 is a noisy year.
- `src/lights_after_road.py`: event study, roads completed 2013-14 vs sanctioned 2011-13 but not
  done by 2015; village + district-by-year FE; flat before the road, rises slowly to about +0.01
  log(1+lights) after 8-12 years -> results/lights_after_road.csv.
- `src/build_tracker.py` -> docs/data (207,576 villages with survey data, 190 districts).
  Left behind = poorest 30% of district in 2012 (SECC poverty rate) and slowest 30% to brighten
  (2012-14 vs 2023-25), villages of 100+ people: 19,827 flagged.
- Website in docs/ (GitHub Pages): index.html (tracker, CSV download), trust.html (findings, slider).
