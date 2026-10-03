"""Night-time brightness of every village, year by year, from 2012 to the latest full year.

India's last village-level wealth data is the 2011-12 survey (SECC). Night lights are
recorded every month by the VIIRS satellite, so they show how each village has changed
since then. This is the "change since the survey" signal for the village progress tracker.

For each village: the median of the monthly VIIRS composites (VCMCFG, avg_rad) in each
calendar year, averaged over a 2 km x 2 km square around the village centre. One
consistent product is used for every year so years can be compared. Months with no
valid data (e.g. summer stray light in the north) are skipped by the median.

Run:    python src/export_lights_timeseries.py            (all ~217,000 villages)
        python src/export_lights_timeseries.py --limit 2000   (quick test)
Output: data/processed/lights_by_year.csv  (shrid2, lights_2012, ..., lights_YYYY)
Re-running continues where it stopped.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd

PROJECT = "satellite-policy-india"
VILLAGES = Path("data/processed/villages.csv.gz")
OUT = Path("data/processed/lights_by_year.csv")
PARTS = Path("data/processed/lights_parts")
HALF_SIDE = 1000  # metres
BATCH = 1000
COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"


def last_full_year():
    latest = ee.Date(ee.ImageCollection(COLLECTION).aggregate_max("system:time_start")).get("year")
    year = latest.getInfo()
    # only use a year if December is in the collection
    dec = ee.ImageCollection(COLLECTION).filterDate(f"{year}-12-01", f"{year + 1}-01-01").size().getInfo()
    return year if dec else year - 1


def yearly_image(years):
    bands = []
    for y in years:
        img = (ee.ImageCollection(COLLECTION).filterDate(f"{y}-01-01", f"{y + 1}-01-01")
               .select("avg_rad").median().rename(f"lights_{y}"))
        bands.append(img)
    return ee.Image.cat(bands)


def run_batch(image, chunk, part_path):
    features = [ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]).buffer(HALF_SIDE).bounds(),
                           {"shrid2": r.shrid2}) for r in chunk.itertuples()]
    for attempt in range(6):
        try:
            result = image.reduceRegions(ee.FeatureCollection(features), ee.Reducer.mean(), scale=460)
            rows = [f["properties"] for f in result.getInfo()["features"]]
            pd.DataFrame(rows).to_csv(part_path, index=False)
            return len(rows)
        except Exception as err:
            if attempt == 5:
                raise
            print(f"  retry ({err})")
            time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    years = list(range(2012, last_full_year() + 1))
    print(f"Years: {years[0]}-{years[-1]}")
    image = yearly_image(years)

    villages = pd.read_csv(VILLAGES, dtype={"shrid2": str}, usecols=["shrid2", "latitude", "longitude"])
    villages = villages.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if args.limit:
        villages = villages.head(args.limit)
    PARTS.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, start in enumerate(range(0, len(villages), BATCH)):
        part = PARTS / f"part_{i:04d}.csv"
        if not part.exists():
            jobs.append((villages.iloc[start:start + BATCH], part))
    total = -(-len(villages) // BATCH)
    print(f"Villages: {len(villages):,} in {total} batches; {total - len(jobs)} already done")

    done, start_time = total - len(jobs), time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_batch, image, chunk, part) for chunk, part in jobs]
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 10 == 0 or done == total:
                rate = (time.time() - start_time) / max(done - (total - len(jobs)), 1)
                print(f"  {done}/{total} batches, about {rate * (total - done) / 60:.0f} min left")

    out = pd.concat([pd.read_csv(p, dtype={"shrid2": str}) for p in sorted(PARTS.glob("part_*.csv"))])
    cols = ["shrid2"] + [f"lights_{y}" for y in years]
    out = out.reindex(columns=cols).drop_duplicates("shrid2")
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {len(out):,} villages")
    print(out[cols[1:]].median().round(3).to_string())


if __name__ == "__main__":
    main()
