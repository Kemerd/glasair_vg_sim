# -*- coding: utf-8 -*-
"""
Live convergence + flow-field + machine-utilization dashboard for the
running validation sweep.

One auto-refreshing matplotlib window, five panels:

    [1] Cl vs iteration for every case with force history (active cases
        solid, finished faded) - the "are we converging" view.
    [2] Residual history (semilog) of the most recently active case, with
        the 1e-5 spec gate marked.
    [3] Per-core CPU utilization bars - the solver ranks live here (the
        flow solver is CPU-only by design; spec section 2 reserves the GPU
        for post-processing and rendering).
    [4] Status panel: sweep progress (converged / failed / solving /
        pending), GPU utilization/VRAM/temperature (the GPU load shown is
        this dashboard's own PyVista rendering plus anything else on the
        card), and field-render bookkeeping.
    [5] Velocity-magnitude field around the airfoil, rendered off-screen by
        PyVista (GPU) from the newest snapshot available - the steady
        solver writes fields every 1000 iterations (purgeWrite 2), so this
        updates a few times per case mid-solve.

All data access is read-only and crash-tolerant: partially-written solver
files just mean "try again next refresh". The only thing ever written into
a case directory is the tiny 'view.foam' stub the OpenFOAM reader needs.

Run:  python scripts/live_dashboard.py [--case-root cases/validation]
      [--interval 3] [--field-interval 15]
Close the window (or Ctrl+C in the console) to stop.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

# Optional hardware/render dependencies degrade panel-by-panel, never fatally.
try:
    import pyvista as pv
    _HAVE_PV = True
except Exception:                                      # noqa: BLE001
    _HAVE_PV = False
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:                                      # noqa: BLE001
    _HAVE_PSUTIL = False

# A case counts as "active" when any postProcessing file changed within this
# window; drives line styling, the residual panel, and the status counts.
ACTIVE_WINDOW_S = 90.0


# =============================================================================
#  Case discovery, freshness, and sweep status
# =============================================================================
def discover_cases(case_root: Path) -> List[Path]:
    """Case dirs that exist under the sweep root, oldest first."""
    return sorted(d for d in case_root.glob("val2d_*") if d.is_dir())


def last_activity(case_dir: Path) -> float:
    """Newest mtime under postProcessing/ (0.0 when nothing is there yet)."""
    newest = 0.0
    for p in (case_dir / "postProcessing").rglob("*.dat"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass                                       # file rotating mid-stat
    return newest


def sweep_status(cases: List[Path], activity: Dict[Path, float],
                 now: float) -> Dict[str, int]:
    """Counts per case state from the driver's m1_status.json markers.

    A case currently writing postProcessing data is 'solving' regardless of
    any stale marker (the driver rewrites markers only at case end).
    """
    counts = {"converged": 0, "failed": 0, "solving": 0, "pending": 0}
    for c in cases:
        if now - activity.get(c, 0.0) < ACTIVE_WINDOW_S:
            counts["solving"] += 1
            continue
        marker = c / "m1_status.json"
        state = "pending"
        if marker.is_file():
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
                state = "converged" if data.get("converged") else "failed"
            except Exception:                          # noqa: BLE001
                state = "pending"                      # marker mid-write
        counts[state] += 1
    return counts


# =============================================================================
#  Hardware utilization (CPU via psutil, GPU via nvidia-smi)
# =============================================================================
def gpu_stats() -> Optional[Dict[str, float]]:
    """One-shot nvidia-smi query; None when no NVIDIA tooling answers."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        util, used, total, temp = [float(v) for v in
                                   out.stdout.strip().split(",")]
        return {"util": util, "vram_used": used, "vram_total": total,
                "temp": temp}
    except Exception:                                  # noqa: BLE001
        return None


# =============================================================================
#  Field rendering (PyVista, off-screen -> RGB array for imshow)
# =============================================================================
def latest_field_source(case_dir: Path) -> Optional[Tuple[str, float]]:
    """Newest readable field snapshot: reconstructed preferred, else the
    decomposed processor0/ tree (mid-solve). None while only 0/ exists."""
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
        # Snap to the closest time the reader actually offers - a snapshot
        # can be purged (purgeWrite 2) between discovery and read.
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
        # Tight orthographic close-up framing ~2.4 chords around the section.
        plotter.camera_position = [(0.5, 0.0, 5.0), (0.5, 0.0, 0.05), (0, 1, 0)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.75
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img
    except Exception:                                  # noqa: BLE001
        # Mid-write time dirs and purge races land here by design: the old
        # image stays on screen until the next refresh succeeds.
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
                     figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.55, 1.25],
                          hspace=0.42, wspace=0.22)
    ax_cl = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[0, 1])
    ax_cpu = fig.add_subplot(gs[1, 0])
    ax_stat = fig.add_subplot(gs[1, 1])
    ax_field = fig.add_subplot(gs[2, :])
    ax_field.set_axis_off()

    if _HAVE_PSUTIL:
        psutil.cpu_percent(percpu=True)                # prime the sampler

    last_field_render = 0.0
    field_img: Optional[np.ndarray] = None
    field_label = ""
    render_ms = 0.0
    render_count = 0

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
        ax_cl.set_title(f"lift convergence ({len(active)} case(s) solving)")
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

        # ---- panel 3: per-core CPU bars ----------------------------------
        ax_cpu.clear()
        if _HAVE_PSUTIL:
            per_core = psutil.cpu_percent(percpu=True)
            total = sum(per_core) / max(1, len(per_core))
            colors = ["#27c93f" if v < 60 else
                      "#f5a623" if v < 90 else "#ff5566" for v in per_core]
            ax_cpu.bar(range(len(per_core)), per_core, color=colors, width=0.8)
            ax_cpu.set_ylim(0, 100)
            ax_cpu.set_title(f"CPU per logical core - total {total:.0f}% "
                             f"(solver is CPU-side: 2 cases x 8 MPI ranks)")
            ax_cpu.set_xlabel("core")
            ax_cpu.grid(True, axis="y", lw=0.4)
        else:
            ax_cpu.set_title("psutil unavailable - CPU panel disabled")

        # ---- panel 4: sweep / GPU / render status ------------------------
        ax_stat.clear()
        ax_stat.set_axis_off()
        counts = sweep_status(cases, activity, now)
        gpu = gpu_stats()
        lines = [
            f"sweep   : {counts['converged']} converged | "
            f"{counts['failed']} failed | {counts['solving']} solving | "
            f"{counts['pending']} pending  (of {len(cases)})",
        ]
        if gpu is not None:
            lines.append(
                f"GPU     : {gpu['util']:.0f}% util | "
                f"{gpu['vram_used']:.0f}/{gpu['vram_total']:.0f} MiB VRAM | "
                f"{gpu['temp']:.0f} C   (renders this dashboard + post-proc;"
                f" the solver is CPU-only by spec)")
        else:
            lines.append("GPU     : nvidia-smi not answering")
        if render_count:
            lines.append(f"renders : {render_count} done | last "
                         f"{render_ms:.0f} ms | next in <= "
                         f"{max(0.0, args.field_interval - (now - last_field_render)):.0f} s")
        else:
            lines.append("renders : waiting for the first field snapshot")
        ax_stat.text(0.02, 0.92, "\n\n".join(lines), va="top", ha="left",
                     family="monospace", fontsize=10, color="#e8eaf0",
                     transform=ax_stat.transAxes)

        # ---- panel 5: newest flow field ----------------------------------
        if now - last_field_render >= args.field_interval:
            last_field_render = now
            pick = target or (cases[-1] if cases else None)
            if pick is not None:
                src = latest_field_source(pick)
                if src is not None:
                    t0 = _time.time()
                    img = render_field(pick, src[0], src[1])
                    if img is not None:
                        render_ms = (_time.time() - t0) * 1000.0
                        render_count += 1
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
