"""Which development split the model-development scripts work on.

The environment variable GL_FOLD selects it; subprocesses started by the run scripts inherit it.
  unset or 2023  the main development split: train before 1 May 2023 (88 months 2016-01 .. 2023-04), evaluate on
                 1 May - 19 October 2023; preparations public/data/v2, runs public/runs/v2, its own logs and
                 evaluation folders, the Assessor refitted on the split in ccao/runs/2025_dev.
  2022           fold 2: the same one year earlier: train before 1 May 2022 (76 months 2016-01 .. 2022-04), evaluate on
                 1 May - 19 October 2022, nothing from 20 October 2022 read (as the held-out start of the 2023
                 split); its own folders (preparations public/data/v2/fold2, runs public/runs/v2/fold2, and logs and
                 evaluations in fold2 subfolders) and its own Assessor refit ccao/runs/2025_dev2022. Run names and
                 preparation names are those of the 2023 split, inside those folders.
COUNTS = (filtered sales before the held-out start, training, evaluation), asserted by loop_prep.py.
"""
import os

FOLD = os.environ.get("GL_FOLD", "2023")
assert FOLD in ("2023", "2022"), f"GL_FOLD={FOLD!r}: only 2023 (default) or 2022"
if FOLD == "2023":
    DEV_START, HELD_OUT_START, NT = "2023-05-01", "2023-10-20", 88
    COUNTS = (343466, 324273, 19193)
    DATA, RUNS, LOGS, EVAL = "public/data/v2", "public/runs/v2", "notes/v2/logs", "notes/v2/loop_eval"
    ASSESSOR = "ccao/runs/2025_dev"
    LABEL = "1 May - 19 Oct 2023"
else:
    DEV_START, HELD_OUT_START, NT = "2022-05-01", "2022-10-20", 76
    COUNTS = (307714, 285283, 22431)
    DATA, RUNS, LOGS, EVAL = ("public/data/v2/fold2", "public/runs/v2/fold2", "notes/v2/logs/fold2",
                              "notes/v2/loop_eval/fold2")
    ASSESSOR = "ccao/runs/2025_dev2022"
    LABEL = "1 May - 19 Oct 2022"
