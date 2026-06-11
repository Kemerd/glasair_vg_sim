# -*- coding: utf-8 -*-
"""
Live convergence + flow-field dashboard for the running validation sweep.

One auto-refreshing matplotlib window with three panels:

    [1] Cl vs iteration for every case that has force history (active cases
        drawn solid, finished ones faded) - the "are we converging" view.
    [2] Residual history (semilog) for the most recently active case.
    [3] Velocity-magnitude field around the airfoil, rendered off-screen by
        PyVista (GPU) from the newest field snapshot it can find - the
        steady solver writes fields every 1000 iterations (purgeWrite 2),
        so this picture updates a few times per case while it solves.

Data access is read-only and crash-tolerant: every panel reads whatever the
solver has flushed so far through the same parsers the convergence gate uses
(scripts/parse_forces), so a partially-written line or a mid-write time
directory just means "try again on the next refresh", never a crash.

Run:  python scripts/live_dashboard.py [--case-root cases/validation]
      [--interval 3] [--field-interval 15]
Close the window (or Ctrl+C in the console) to stop. The dashboard never
writes into case directories except a tiny ParaView-style 'view.foam' stub
file that the OpenFOAM reader requires to locate a case.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scripts.parse_forces import load_force_coeffs, load_residuals

# PyVista is optional at import time so the convergence panels still work on
# a machine without working OpenGL; the field panel then shows a notice.
try:
    import pyvista as pv
    _HAVE_PV = True
except Exception:                                      # noqa: BLE001
    _HAVE_PV = False

# A case counts as "active" when any of its postProcessing files changed
# within this window; drives line styling and which residuals are shown.
ACTIVE_WINDOW_S = 90.0


# =============================================================================
#  Case discovery and freshness
# =============================================================================
def discover_cases(case_root: Path) -> List[Path]:
    """Case dirs that have started producing force history, oldest first."""
    out = []
    for d in sorted(case_root.glob("val2d_*")):
        if (d / "postProcessing").is_dir():
            out.append(d)
    return out


def last_activity(case_dir: Path) -> float:
    """Newest mtime under postProcessing/ (0.0 when nothing is there yet)."""
    newest = 0.0
    for p in (case_dir / "postProcessing").rglob("*.dat"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass                                       # file rotating mid-stat
    return newest


# =============================================================================
#  Field rendering (PyVista, off-screen -> RGB array for imshow)
# =============================================================================
def latest_field_source(case_dir: Path) -> Optional[Tuple[str, float]]:
    """Newest readable field snapshot for a case.

    Prefers the reconstructed case when a nonzero time dir exists at the
    case root (finished cases), else falls back to the decomposed
    processor0/ tree (mid-solve snapshots). Returns (case_type, time) or
    None when only the initial 0/ state exists - rendering the untouched
    initial field would just show a uniform freestream.
    """
    def newest_time(root: Path) -> float:
        best = 0.0
        for d in root.iterdir() if root.is_dir() else []:
            try:
                t = float(d.name)
            except ValueError:
                continue
            if d.is_dir() and t > best:
                best = t
        return best

    t_recon = newest_time(case_dir)
    if t_recon > 0.0:
        return ("reconstructed", t_recon)
    t_dec = newest_time(case_dir / "processor0")
    if t_dec > 0.0:
        return ("decomposed", t_dec)
    return None


def render_field(case_dir: Path, kind: str, t: float,
                 size: Tuple[int, int] = (1100, 620)) -> Optional[np.ndarray]:
    """Off-screen GPU render of |U| at time t; returns an RGB image array."""
    if not _HAVE_PV:
        return None
    stub = case_dir / "view.foam"
    if not stub.exists():
        stub.touch()                                   # reader needs the stub
    try:
        reader = pv.POpenFOAMReader(str(stub))
        reader.case_type = kind
        # Snap to the closest available time value the reader actually has -
        # a snapshot can be purged (purgeWrite 2) between discovery and read.
        times = [tv for tv in reader.time_values if tv > 0.0]
        if not times:
            return None
        reader.set_active_time_value(min(times, key=lambda tv: abs(tv - t)))
        blocks = reader.read()
        mesh = blocks["internalMesh"]
        u = mesh.point_data.get("U", None)
        if u is None:                                  # fall back to cell data
            mesh = mesh.cell_data_to_point_data()
            u = mesh.point_data.get("U", None)
        if u is None:
            return None
        mesh.point_data["Umag"] = np.linalg.norm(np.asarray(u), axis=1)

        plotter = pv.Plotter(off_screen=True, window_size=list(size))
        plotter.add_mesh(mesh, scalars="Umag", cmap="turbo",
                         show_scalar_bar=True,
                         scalar_bar_args={"title": "|U| m/s", "color": "white"})
        plotter.set_background("#101218")
        # Tight orthographic close-up on the unit-chord airfoil: focal point
        # at mid-chord, parallel scale chosen to frame roughly 2.4 chords.
        plotter.camera_position = [(0.5, 0.0, 5.0), (0.5, 0.0, 0.05), (0, 1, 0)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.75
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img
    except Exception:                                  # noqa: BLE001
        # Mid-write time dirs and purge races land here by design: the old
        # image simply stays on screen until the next refresh succeeds.
        return None


# =============================================================================
#  Dashboard loop
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="live validation dashboard")
    ap.add_argument("--case-root", default=str(REPO / "cases" / "validation"))
    ap.add_argument("--interval", type=float, default=3.0,
                    help="convergence-panel refresh seconds (default 3)")
    ap.add_argument("--field-interval", type=float, default=15.0,
                    help="flow-field re-render seconds (default 15)")
    args = ap.parse_args()
    case_root = Path(args.case_root)

    matplotlib.rcParams.update({
        "figure.facecolor": "#101218", "axes.facecolor": "#161a22",
        "axes.edgecolor": "#3a4150", "axes.labelcolor": "#e8eaf0",
        "text.color": "#e8eaf0", "xtick.color": "#aab0bf",
        "ytick.color": "#aab0bf", "grid.color": "#2a2f3a",
    })
    fig = plt.figure("Glasair VG study - live validation dashboard",
                     figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35],
                          hspace=0.32, wspace=0.22)
    ax_cl = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[0, 1])
    ax_field = fig.add_subplot(gs[1, :])
    ax_field.set_axis_off()

    last_field_render = 0.0
    field_img: Optional[np.ndarray] = None
    field_label = ""

    plt.ion()
    fig.show()
    print("dashboard: watching", case_root, "- close the window to stop")

    while plt.fignum_exists(fig.number):
        now = _time.time()
        cases = discover_cases(case_root)
        activity = {c: last_activity(c) for c in cases}
        active = [c for c in cases if now - activity.get(c, 0) < ACTIVE_WINDOW_S]

        # ---- panel 1: Cl histories -------------------------------------
        ax_cl.clear()
        for c in cases:
            try:
                fh = load_force_coeffs(c)
            except Exception:                          # noqa: BLE001
                continue                               # nothing flushed yet
            if len(fh) < 2:
                continue
            is_live = c in active
            label = c.name.replace("val2d_", "").replace("_re3e6_lvl0", "")
            ax_cl.plot(fh.time, fh.cl,
                       lw=1.8 if is_live else 0.9,
                       alpha=1.0 if is_live else 0.45,
                       label=label if is_live else None)
        ax_cl.set_xlabel("iteration")
        ax_cl.set_ylabel("Cl")
        ax_cl.set_title("lift coefficient convergence "
                        f"({len(active)} case(s) solving)")
        ax_cl.grid(True, lw=0.4)
        if active:
            ax_cl.legend(loc="lower right", fontsize=8, framealpha=0.2)

        # ---- panel 2: residuals of the freshest active case -------------
        ax_res.clear()
        target = max(active, key=lambda c: activity[c]) if active else None
        if target is not None:
            try:
                rh = load_residuals(target)
                for name, vals in sorted(rh.fields.items()):
                    ax_res.semilogy(rh.time, np.maximum(vals, 1e-16),
                                    lw=1.1, label=name)
                ax_res.axhline(1e-5, color="#ff5566", lw=0.9, ls="--")
                ax_res.legend(loc="upper right", fontsize=8, framealpha=0.2,
                              ncol=2)
            except Exception:                          # noqa: BLE001
                pass
            ax_res.set_title(f"residuals - {target.name} (gate 1e-5 dashed)")
        else:
            ax_res.set_title("residuals - waiting for an active case")
        ax_res.set_xlabel("iteration")
        ax_res.grid(True, lw=0.4)

        # ---- panel 3: newest flow field (rendered every field-interval) --
        if now - last_field_render >= args.field_interval:
            last_field_render = now
            pick = target or (cases[-1] if cases else None)
            if pick is not None:
                src = latest_field_source(pick)
                if src is not None:
                    img = render_field(pick, src[0], src[1])
                    if img is not None:
                        field_img = img
                        field_label = (f"{pick.name}   |U| at iter "
                                       f"{src[1]:g} ({src[0]})")
        if field_img is not None:
            ax_field.clear()
            ax_field.set_axis_off()
            ax_field.imshow(field_img)
            ax_field.set_title(field_label, fontsize=11)
        elif not _HAVE_PV:
            ax_field.set_title("PyVista unavailable - field panel disabled")
        else:
            ax_field.set_title("waiting for the first field snapshot "
                               "(written every 1000 iterations)...")

        fig.canvas.draw_idle()
        # plt.pause keeps the GUI event loop alive between refreshes.
        plt.pause(args.interval)

    print("dashboard: window closed, exiting")


if __name__ == "__main__":
    main()
