#!/usr/bin/env python
"""Queue the same gated round for Sports after the first-round GPU is free."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_NAME = "20260923_gated_sports"
LOG = HERE / "runs" / RUN_NAME
LOG.mkdir(parents=True, exist_ok=True)
(LOG / "queue.json").write_text(json.dumps({"queued_for": "20260922_r1/main/Sports",
                                             "gpu": 2, "run_name": RUN_NAME}, indent=2))
with (LOG / "queue.log").open("w") as out:
    out.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + " queued\n")
    while any("run_experiment.py --dataset Sports --variant main --gpu 2 --run-name 20260922_r1" in line
              for line in os.popen("ps -eo args")):
        time.sleep(60)
    cmd = [sys.executable, "-u", str(HERE / "run_experiment.py"), "--dataset", "Sports",
           "--variant", "gated_margin", "--gpu", "2", "--run-name", RUN_NAME]
    out.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + " launching: " + " ".join(cmd) + "\n")
    out.flush()
    os.execvpe(cmd[0], cmd, os.environ.copy())
