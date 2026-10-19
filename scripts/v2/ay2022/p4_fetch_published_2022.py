# 2022 version of scripts/v2/p4_fetch_published_2023.py: the same code with the 2022 assessment year's dates, file names and
# north-suburb labels, and the differences listed in REPRODUCE.md ("What to expect").
"""Download the Assessor's published 2022 assessed values for the north triad (open data uzyt-m557).

Reference for the loose external check of the rebuilt 2022 run (p4_external_check.py).
Usage (repo root): .venv/bin/python -I public/scripts/v2/ay2022/p4_fetch_published_2022.py public/data/v2/ay2022
Writes p8_published_2022_north.parquet and p8_published_2022_north.json (query, time, rows, sha256).
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
SOUTH = ["10", "16", "17", "18", "20", "22", "23", "24", "25", "26", "29", "35", "38"]  # the north triad (variable name kept from the 2023 script)
WHERE = "year = '2022' AND township_code in (" + ", ".join(f"'{t}'" for t in SOUTH) + ")"
SELECT = "pin, year, class, township_code, nbhd, mailed_bldg, mailed_land, mailed_tot, mailed_hie, certified_tot, board_tot"
PAGE = 50000
out = Path(sys.argv[1])
rows = []
for town in SOUTH:  # one township at a time (the whole-triad query timed out)
    off = 0
    while True:
        q = urllib.parse.urlencode({"$select": SELECT, "$where": f"year = '2022' AND township_code = '{town}'",
                                    "$order": "pin", "$limit": PAGE, "$offset": off})
        with urllib.request.urlopen(f"{URL}?{q}", timeout=600) as r:
            page = json.load(r)
        rows += page
        print(f"township {town} offset {off}: {len(page)} rows", flush=True)
        if len(page) < PAGE:
            break
        off += PAGE
df = pd.DataFrame(rows)
f = out / "p8_published_2022_north.parquet"
df.to_parquet(f, index=False)
meta = {"dataset": "Cook County Assessor, Assessed Values (uzyt-m557)", "url": URL, "select": SELECT, "where": WHERE,
        "order": "pin", "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": len(df),
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
(out / "p8_published_2022_north.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
