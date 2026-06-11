# -*- coding: utf-8 -*-
"""
Babysitter for the owner's conditional order: when Act III completes, launch
Act IV -- but only if that happens before the 09:30 deadline.

Trigger logic, polled every 60 s:
  * primary:  'act III complete' appears in suite3.log -> normal completion
  * fallback: no run_suite3.py python process remains AND suite3.log has
              been silent for 15+ minutes -> Act III died; still chain so
              the night is not wasted (Act IV cases are independent)
Deadline: if the trigger fires at or after 09:30 local, log and exit
without launching (the owner wants the machine free by then).
"""
from __future__ import annotations

import datetime
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG3 = HERE / "results" / "suite" / "suite3.log"
LOG = HERE / "results" / "suite" / "chain.log"
DEADLINE = datetime.time(9, 30)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def suite3_running() -> bool:
    """True while a python process with run_suite3 in its command line exists.

    FAIL-SAFE DIRECTION MATTERS (learned at 05:15 the hard way): a hiccuped
    or empty query result previously parsed as '0 processes' and prematurely
    fired Act IV while Act III was mid-case. Any failure to obtain a clean
    integer now reads as 'still running'.
    """
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'run_suite3' }).Count"],
        capture_output=True, text=True)
    s = out.stdout.strip()
    if out.returncode != 0 or s == "":
        return True                                    # query failed -> assume alive
    try:
        return int(s) > 0
    except ValueError:
        return True                                    # garbage output -> assume alive


def main() -> int:
    log("chain watcher armed (v2, fail-safe): Act IV launches when Act III ends (deadline 09:30)")
    stale_polls = 0
    while True:
        time.sleep(60)
        done = LOG3.exists() and "act III complete" in LOG3.read_text(
            encoding="utf-8", errors="replace")
        stale_now = (LOG3.exists()
                     and time.time() - LOG3.stat().st_mtime > 25 * 60
                     and not suite3_running())
        # Two consecutive stale verdicts required: one poll can lie (slow WMI
        # query, quiet long case). The 25 min silence threshold also exceeds
        # the longest expected case runtime at u=0.075.
        stale_polls = stale_polls + 1 if stale_now else 0
        if not (done or stale_polls >= 2):
            continue
        now = datetime.datetime.now().time()
        if now >= DEADLINE:
            log(f"Act III ended ({'complete' if done else 'stalled'}) at "
                f"{now} -- past the 09:30 deadline, Act IV NOT launched")
            return 0
        log(f"Act III ended ({'complete' if done else 'stalled'}) -- launching Act IV")
        subprocess.Popen([sys.executable, str(HERE / "run_suite4.py")],
                         cwd=str(HERE.parents[1]))
        return 0


if __name__ == "__main__":
    sys.exit(main())
