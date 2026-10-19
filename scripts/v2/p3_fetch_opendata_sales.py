"""Download the Assessor's open-data Parcel Sales (Socrata wvhk-k5uv) with sale dates in 2023.

Used only for the document-level reconciliation of the 2023 outcome source (the 2025 model input). All
classes are fetched; filtering is done in p3_eligibility.py. Prices are stored, never summarised here.

Usage (repo root): .venv/bin/python -I public/scripts/v2/p3_fetch_opendata_sales.py public/data/v2
Writes p3_opendata_sales_2023.parquet and p3_opendata_sales_2023.json (query, time, rows, sha256).
"""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.json"
WHERE = "sale_date >= '2023-01-01T00:00:00' AND sale_date < '2024-01-01T00:00:00'"
PAGE = 50000
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

rows, off = [], 0
while True:
    q = urllib.parse.urlencode({"$where": WHERE, "$order": "row_id", "$limit": PAGE, "$offset": off})
    with urllib.request.urlopen(f"{URL}?{q}", timeout=300) as r:
        page = json.load(r)
    rows += page
    print(f"offset {off}: {len(page)} rows", flush=True)
    if len(page) < PAGE:
        break
    off += PAGE
df = pd.DataFrame(rows)
f = out / "p3_opendata_sales_2023.parquet"
df.to_parquet(f, index=False)
meta = {"dataset": "Cook County Assessor, Parcel Sales (wvhk-k5uv)", "url": URL, "where": WHERE,
        "order": "row_id", "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": len(df),
        "columns": list(df.columns), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
(out / "p3_opendata_sales_2023.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
