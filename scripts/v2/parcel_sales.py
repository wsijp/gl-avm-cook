"""Download the Cook County Assessor's Parcel Sales table (open data, Socrata dataset wvhk-k5uv, sales
recorded from 1999) for a set of townships, for complete sale histories (dates only; buyer and seller names are not
requested).

Every class is downloaded; the history filter (class, deed type, multi-sale and sale-filter flags) is applied by the
history code, not here. Written to public/data/v2/parcel_sales/sales_t<codes>.parquet (all columns as strings) with a
JSON manifest (query, row count, download time, sha256).

Usage (repo root): .venv/bin/python -I public/scripts/v2/parcel_sales.py --towns 77 73 76 74 31 11 [--out <folder>]
(--out, added 2026-10-24 for the release's smoke test: write elsewhere than public/data/v2/parcel_sales)
"""
import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "public/data/v2/parcel_sales"
URL = "https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.csv"
PAGE = 50_000
COLS = ["pin", "year", "township_code", "nbhd", "class", "sale_date", "is_mydec_date", "sale_price", "doc_no",
        "deed_type", "mydec_deed_type", "is_multisale", "num_parcels_sale", "sale_type",
        "sale_filter_same_sale_within_365", "sale_filter_less_than_10k", "sale_filter_deed_type", "row_id"]


def fetch(where):
    frames, offset = [], 0
    while True:
        q = {"$select": ",".join(COLS), "$where": where, "$order": "row_id", "$limit": PAGE, "$offset": offset}
        url = URL + "?" + urllib.parse.urlencode(q)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    text = r.read().decode("utf-8")
                break
            except Exception as e:  # network hiccup: back off and retry
                print(f"  retry {attempt + 1}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"failed after 5 attempts: {url}")
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        frames.append(df)
        print(f"  offset {offset:>9,}: {len(df):,} rows", flush=True)
        if len(df) < PAGE:
            break
        offset += PAGE
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    where = "township_code in(" + ",".join(f"'{t}'" for t in a.towns) + ")"
    d = fetch(where)
    assert d["row_id"].is_unique, "row_id must be unique (stable paging)"
    f = a.out / f"sales_t{'-'.join(a.towns)}.parquet"
    d.to_parquet(f, index=False)
    man = dict(dataset="Assessor - Parcel Sales", socrata_id="wvhk-k5uv", where=where, select=COLS, rows=len(d),
               rows_by_town=d["township_code"].value_counts().to_dict(),
               sale_date_min=d["sale_date"].min(), sale_date_max=d["sale_date"].max(),
               downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               sha256=hashlib.sha256(f.read_bytes()).hexdigest(), note="buyer_name and seller_name not requested")
    f.with_suffix(".json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
