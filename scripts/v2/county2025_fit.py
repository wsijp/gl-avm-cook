"""Sale-prediction test: fit the published model P to one township prepared by county2025_prep.py.

fit_struct.py unchanged, with one difference: 94 training months (2016-01 .. 2023-10) instead of 108. fit_struct.py
reads the month count from fit_v2.NT when it is imported, so this wrapper sets that constant first; the published
scripts stay byte-identical. Arguments are fit_struct.py's.

Usage (repo root): .venv/bin/python -I public/scripts/v2/county2025_fit.py --town 72 --model P \
                   --prep public/data/v2/prep_county2025 --name county2025_P_t72
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_v2  # noqa: E402

fit_v2.NT = 94
import fit_struct  # noqa: E402

assert fit_struct.NT == 94
if __name__ == "__main__":
    fit_struct.main()
