"""Night-time brightness around each village, as an extra input for the wealth model.

Daytime images show roofs, fields and roads; night lights show electricity use, which
tracks income closely. For every village with a Landsat tile, this takes the average
brightness over the same 2.9 km x 2.9 km square the tile covers, from two sources:

  viirs_2012   VIIRS monthly composites, Apr-Dec 2012 (nW/cm2/sr), ~460 m pixels
  dmsp_2011    DMSP-OLS stable lights, 2011 (0-63 scale), ~1 km pixels

Both are from around the time of the 2011-12 survey. Asks Earth Engine for one
summary number per village, so it takes minutes, not hours.

Run: python src/export_nightlights.py
Output: data/processed/nightlights.csv (shrid2, viirs_2012, dmsp_2011)
"""
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

PROJECT = "satellite-policy-india"
VILLAGES = Path("data/processed/villages.csv.gz")
TILES = Path("data/tiles")
OUT = Path("data/processed/nightlights.csv")
HALF_SIDE = 96 * 30 / 2   # metres; matches the Landsat tile
BATCH = 1000


def lights_image():
    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
             .filterDate("2012-04-01", "2013-01-01").select("avg_rad")
             .mean().rename("viirs_2012"))
    dmsp = (ee.Image("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F182011")
            .select("stable_lights").rename("dmsp_2011"))
    return viirs.addBands(dmsp)


def batch_means(image, rows):
    features = [ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(HALF_SIDE).bounds(),
                           {"shrid2": r.shrid2}) for r in rows.itertuples()]
    result = image.reduceRegions(ee.FeatureCollection(features), ee.Reducer.mean(), scale=460)
    return [f["properties"] for f in result.getInfo()["features"]]


def main():
    ee.Initialize(project=PROJECT)
    ids = []
    for f in sorted(TILES.glob("shard_*.npz")):
        ids.extend(np.load(f, allow_pickle=True)["shrid2"].tolist())
    villages = pd.read_csv(VILLAGES, dtype={"shrid2": str}, usecols=["shrid2", "latitude", "longitude"])
    todo = villages[villages["shrid2"].isin(set(ids))].reset_index(drop=True)
    print(f"Villages with tiles: {len(todo):,}")

    image, rows = lights_image(), []
    for start in range(0, len(todo), BATCH):
        chunk = todo.iloc[start:start + BATCH]
        for attempt in range(5):
            try:
                rows.extend(batch_means(image, chunk))
                break
            except Exception as err:
                print(f"  retrying after error: {err}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("Earth Engine kept failing; run again later")
        print(f"  {min(start + BATCH, len(todo)):,}/{len(todo):,}")

    out = pd.DataFrame(rows).reindex(columns=["shrid2", "viirs_2012", "dmsp_2011"])
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {len(out):,} villages")
    print(out[["viirs_2012", "dmsp_2011"]].describe().round(2))


if __name__ == "__main__":
    main()
