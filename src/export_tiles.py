"""Download a small Landsat image around each sampled village from Google Earth Engine.

For every village we cut a 96 x 96 pixel tile (30 m pixels, about 2.9 km across)
centred on the village, from a cloud-free composite of Landsat 7 surface reflectance
for the dry seasons around the 2011 census and 2012 SECC (Oct 2010 - Mar 2012).
Six bands: blue, green, red, near-infrared, shortwave-infrared 1 and 2.

Tiles are saved in shards of 1,000 (data/tiles/shard_XXX.npz), each holding
  images: uint8 array (n, 6, 96, 96)   reflectance 0-0.6 scaled to 0-255
  shrid2: the village IDs, in the same order
The script can be stopped and restarted; it skips villages already downloaded.

Sample: every village in the roads-study sample, plus a random set of other
villages from each state (for training the model).

First run:  earthengine authenticate
Test run:   python src/export_tiles.py --limit 40 --preview
Full run:   python src/export_tiles.py
"""
import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

PROJECT = "satellite-policy-india"
VILLAGES = Path("data/processed/villages.csv.gz")
TILES = Path("data/tiles")
SIZE = 96          # pixels per side
PIXEL = 30         # metres per pixel (Landsat)
BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
TRAIN_PER_STATE = 3000
SEED = 2027
SHARD = 1000


def composite():
    """Median of cloud-masked Landsat 7 scenes, dry season Oct 2010 - Mar 2012."""
    def clean(img):
        qa = img.select("QA_PIXEL")
        clear = (qa.bitwiseAnd(1 << 1).eq(0)        # dilated cloud
                 .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
                 .And(qa.bitwiseAnd(1 << 4).eq(0))) # cloud shadow
        return img.select(BANDS).multiply(0.0000275).add(-0.2).updateMask(clear)

    india_centre = ee.Geometry.Rectangle([68, 15, 89, 31])  # covers the six states (lat 15-31 N)
    return (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterBounds(india_centre)
            .filter(ee.Filter.Or(ee.Filter.date("2010-10-01", "2011-04-01"),   # two dry seasons
                                 ee.Filter.date("2011-10-01", "2012-04-01")))
            .map(clean)
            .median()
            .multiply(255 / 0.6).clamp(0, 255).toUint8()
            .unmask(0))


def to_mercator(lat, lon):
    r = 6378137.0
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def fetch(image, lat, lon):
    """Return a (6, SIZE, SIZE) uint8 array centred on (lat, lon)."""
    # Web Mercator stretches distances by 1/cos(lat); correct so pixels are ~30 m on the ground
    step = PIXEL / math.cos(math.radians(lat))
    x, y = to_mercator(lat, lon)
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": SIZE, "height": SIZE},
            "affineTransform": {
                "scaleX": step, "shearX": 0, "translateX": x - step * SIZE / 2,
                "shearY": 0, "scaleY": -step, "translateY": y + step * SIZE / 2,
            },
            "crsCode": "EPSG:3857",
        },
    }
    for attempt in range(5):
        try:
            arr = ee.data.computePixels(request)
            return np.stack([arr[b] for b in BANDS])
        except ee.ee_exception.EEException as err:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)  # back off on rate limits / transient errors


def choose_sample(villages):
    rd = villages[villages["in_rd_sample"] == 1]
    rest = villages[(villages["in_rd_sample"] == 0) & villages["secc_cons_pc_rural"].notna()]
    # shuffle once, then take the first TRAIN_PER_STATE villages of each state
    train = rest.sample(frac=1, random_state=SEED).groupby("state_name").head(TRAIN_PER_STATE)
    return pd.concat([rd, train])


def already_done():
    done = set()
    for f in TILES.glob("shard_*.npz"):
        done.update(np.load(f, allow_pickle=True)["shrid2"].tolist())
    return done


def save_shard(images, ids):
    n = len(list(TILES.glob("shard_*.npz")))
    np.savez_compressed(TILES / f"shard_{n:03d}.npz",
                        images=np.stack(images), shrid2=np.array(ids, dtype=object))


def preview(images, ids):
    """Save true-colour (red, green, blue) previews to check the tiles look right."""
    from PIL import Image
    out = TILES / "preview"
    out.mkdir(exist_ok=True)
    for img, sid in list(zip(images, ids))[:12]:
        rgb = np.stack([img[2], img[1], img[0]], axis=-1).astype(float)
        rgb = np.clip(rgb * 2.5, 0, 255).astype(np.uint8)  # brighten for viewing
        Image.fromarray(rgb).resize((288, 288), Image.NEAREST).save(out / f"{sid}.png")
    print(f"Saved previews to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only download this many (for testing)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--check", action="store_true", help="Print scene count and one tile's values, then stop")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    TILES.mkdir(parents=True, exist_ok=True)

    villages = pd.read_csv(VILLAGES, dtype={"shrid2": str})
    sample = choose_sample(villages)
    done = already_done()
    todo = sample[~sample["shrid2"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Sample: {len(sample):,} villages ({sample['in_rd_sample'].sum():,} roads-study, "
          f"{(sample['in_rd_sample'] == 0).sum():,} training). Already done: {len(done):,}. "
          f"To download now: {len(todo):,}")

    image = composite()
    if args.check:
        scenes = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                  .filterBounds(ee.Geometry.Rectangle([68, 15, 89, 31]))
                  .filter(ee.Filter.Or(ee.Filter.date("2010-10-01", "2011-04-01"),
                                       ee.Filter.date("2011-10-01", "2012-04-01"))))
        print(f"Landsat 7 scenes in the composite: {scenes.size().getInfo():,}")
        r = todo.iloc[0]
        tile = fetch(image, r.latitude, r.longitude)
        print(f"Test tile for {r.shrid2} ({r.latitude:.3f}, {r.longitude:.3f}): "
              f"band means {tile.reshape(6, -1).mean(axis=1).round(1).tolist()}")
        return
    images, ids, failed = [], [], 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, image, r.latitude, r.longitude): r.shrid2
                for r in todo.itertuples()}
        for i, job in enumerate(as_completed(jobs), 1):
            try:
                images.append(job.result())
                ids.append(jobs[job])
            except Exception as err:  # keep going; failed villages are retried next run
                failed += 1
                if failed <= 5:
                    print(f"  failed {jobs[job]}: {err}")
            if len(images) >= SHARD:
                save_shard(images, ids)
                images, ids = [], []
            if i % 200 == 0:
                rate = i / (time.time() - start)
                print(f"  {i:,}/{len(todo):,} done, {rate:.1f}/s, "
                      f"about {(len(todo) - i) / rate / 60:.0f} min left")

    if images:
        if args.preview:
            preview(images, ids)
        blank = np.mean([(img == 0).all(axis=0).mean() for img in images])
        print(f"Average share of empty (no clear image) pixels per tile: {blank:.1%}")
        save_shard(images, ids)
    print(f"Finished. Failed: {failed} (run again to retry them)")


if __name__ == "__main__":
    main()
