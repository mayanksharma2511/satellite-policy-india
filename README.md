# Village Progress Tracker

**Live site: https://mayanksharma2511.github.io/satellite-policy-india/**

India's last village-level data on household wealth is the 2011–12 Socio-Economic and Caste Census (SECC). Anyone choosing villages for a programme today, such as an NGO or a company's CSR team, is working from data that is 14 years old. This project does two things:

1. **A tool.** For about 207,000 villages in six states, it combines each village's 2012 poverty rate with night-time brightness measured by satellite every year from 2012 to 2025. It shortlists villages that were among the poorest in their district in 2012 and are still among the darkest at night today. With the default settings it flags 24,978 villages. Users pick a district and a number of villages they can reach, and get a ranked list. They can view it on a map or as a table, and download it with tehsil names and Google Maps links.
2. **A test of how far satellite estimates can be trusted.** I trained a CNN to estimate village consumption from Landsat images. I then re-ran a published study of India's rural roads programme (Asher & Novosad, *American Economic Review*, 2020) with the satellite estimates in place of the survey, to see whether they reach the same conclusion.

## Main findings

| Question | Answer |
|---|---|
| Does the replication work? | Yes. Effect of a new road on the share of people working in cultivation: −0.141 (95% CI −0.291 to +0.010) against the paper's −0.092. No effect on consumption, as in the paper. |
| How well does the CNN rank villages it never saw? | Within-state rank correlation 0.40 (R² 0.20). In a state left out of training: 0.34 (R² 0.09). A ridge model on image summary statistics: 0.34 and 0.30. |
| Does the satellite version of the roads study agree? | Yes: −1.6% (95% CI −5.7% to +2.5%), the same as the survey's −1.6%. |
| Is the agreement meaningful? | Weakly. Satellite estimates move only 0.21× as much as true consumption (within-state slope), so the satellite study could only detect a road effect of about 28%, against about 14% for the survey. |
| Does a better model fix this? | No. Across five models (20 epochs, adding night lights, simpler models) the shrink factor stayed between 0.09 and 0.22. |
| Can satellite estimates create false findings? | Yes. The ridge model reports a "significant" negative road effect (−3.5%, CI −6.6% to −0.4%) that the survey does not show. |
| Is the model seeing the new roads themselves? | No. Its error does not jump at the cut-off (+0.000, CI −0.016 to +0.017). |
| Does night-light change mean anything? | Villages whose road was completed in 2013–14 brighten slowly afterwards compared with villages still waiting for theirs (flat before, about +0.01 log points after 8–12 years). |

**In short:** satellite data is usable for choosing where to look, as long as the result is treated as a shortlist and checked on the ground. It is risky for judging whether a programme worked.

## Method

- **Data.** SHRUG v2.2 (Development Data Lab): SECC 2012 consumption and poverty, Census 2001 and 2011, and PMGSY road records to 2015, for Chhattisgarh, Gujarat, Madhya Pradesh, Maharashtra, Odisha and Rajasthan.
- **Images.** Landsat 7 surface reflectance, dry seasons 2010–12, cloud-masked median. Six bands, 96×96 pixels (2.9 km) around each village, for 26,040 villages: every village in the roads study plus 1,500 per state for training.
- **Model.** ResNet-18 pretrained on ImageNet with the first layer adapted to six bands. Trained on a free Kaggle GPU. Predictions are cross-fitted by district (5 folds, so no village is predicted by a model that saw it or its neighbours), plus a leave-one-state-out test.
- **Roads study.** Fuzzy regression discontinuity at the PMGSY population cut-offs (500 and 1,000), bandwidth 84, triangular kernel, district × cut-off fixed effects, 2001 controls, robust standard errors. 16,024 villages.
- **Night lights.** VIIRS monthly composites (VCMCFG), yearly median 2012–2025, over a 2 km square around each village. The 2016 values are noisy, so the tracker compares three-year averages.
- **Left behind.** Among the poorest 30% of the district in 2012 (SECC poverty rate) and the darkest 30% today (average of 2023–25). Only villages of 100+ people are counted, because survey rates for tiny hamlets are unreliable. Users can switch to 20% ("most urgent only") or 40% ("wider search").

`NOTES.md` records every run, including the ones that did not help, and the reasons for each decision.

## Repository

```
src/build_village_table.py       join SHRUG modules into one village table; RD running variable
src/export_tiles.py              Landsat tiles from Earth Engine
src/train_cnn.py                 CNN + ridge baseline, cross-fit and leave-one-state-out (Kaggle)
src/evaluate_predictions.py      R², within-state rank correlation
src/rd_roads.py                  roads regression discontinuity, survey vs satellite
src/export_nightlights.py        2011-12 night lights as a model input
src/export_lights_timeseries.py  night lights 2012-2025 for every village
src/lights_after_road.py         event study: brightness after a road
src/build_tracker.py             data files for the website
docs/                            the website (GitHub Pages)
results/                         regression outputs
NOTES.md                         run log and decisions
```

## Reproducing

1. Download the SHRUG modules listed in `src/build_village_table.py` into `data/raw/shrug/`, then run `python src/build_village_table.py`.
2. Set up a free Google Earth Engine project, then run `python src/export_tiles.py --workers 24`, `python src/export_nightlights.py` and `python src/export_lights_timeseries.py`.
3. Upload `data/tiles/`, `data/processed/villages.csv.gz` and `nightlights.csv` to a Kaggle notebook (GPU on), then run `python train_cnn.py --mode all --epochs 8`. Copy the outputs into `outputs/`.
4. Run `python src/evaluate_predictions.py`, `python src/rd_roads.py`, `python src/lights_after_road.py` and `python src/build_tracker.py`.

Python packages: `requirements.txt`. Training also needs PyTorch and torchvision, which are preinstalled on Kaggle. Everything used is free.

## Limits

- Six states and one programme. The results show that satellite evaluation can fail, and by how much here, not that it always does.
- The 2012 poverty rates are SECC-based estimates and are now old. Night lights capture electrification and large changes in activity, not small changes in income.
- The 2 km squares of neighbouring villages overlap, so nearby villages often have similar brightness.
- Road records end in 2015. Some villages have no recorded name.
- The tracker is a prototype that has not yet been checked against conditions on the ground.

## Credits

SHRUG: Asher, Lunt, Matsuura and Novosad (2021), Development Data Lab. Rural roads study: Asher and Novosad (2020), "Rural Roads and Local Economic Development", *American Economic Review*. Landsat: USGS. VIIRS and DMSP night lights: NOAA / Earth Observation Group. Imagery basemap: Esri.
