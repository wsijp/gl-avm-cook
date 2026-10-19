"""Download the Assessor's published 2023 assessed values for the south triad (open data uzyt-m557).

Reference for the loose external check of the rebuilt 2023 run (p4_external_check.py).
Usage (repo root): .venv/bin/python -I public/scripts/v2/p4_fetch_published_2023.py public/data/v2
Writes p4_published_2023_south.parquet and p4_published_2023_south.json (query, time, rows, sha256).
"""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://datacatalog.cookcountyil.gov/resource/uzyt-m557.json"
SOUTH = ["11", "12", "13", "14", "15", "19", "21", "27", "28", "30", "31", "32", "33", "34", "36", "37", "39"]
WHERE = "year = '2023' AND township_code in (" + ", ".join(f"'{t}'" for t in SOUTH) + ")"
SELECT = "pin, year, class, township_code, nbhd, mailed_bldg, mailed_land, mailed_tot, mailed_hie, certified_tot, board_tot"
PAGE = 50000
out = Path(sys.argv[1])
rows = []
for town in SOUTH:  # one township at a time (the whole-triad query timed out)
    off = 0
    while True:
        q = urllib.parse.urlencode({"$select": SELECT, "$where": f"year = '2023' AND township_code = '{town}'",
                                    "$order": "pin", "$limit": PAGE, "$offset": off})
        with urllib.request.urlopen(f"{URL}?{q}", timeout=600) as r:
            page = json.load(r)
        rows += page
        print(f"township {town} offset {off}: {len(page)} rows", flush=True)
        if len(page) < PAGE:
            break
        off += PAGE
df = pd.DataFrame(rows)
f = out / "p4_published_2023_south.parquet"
df.to_parquet(f, index=False)
meta = {"dataset": "Cook County Assessor, Assessed Values (uzyt-m557)", "url": URL, "select": SELECT, "where": WHERE,
        "order": "pin", "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": len(df),
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
(out / "p4_published_2023_south.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
