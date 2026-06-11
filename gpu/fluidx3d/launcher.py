# -*- coding: utf-8 -*-
"""
Glasair GPU tunnel launcher -- a small visual cockpit for the FluidX3D rig.

Pick angle of attack, airspeed (mph, converted to Reynolds internally), and
the VG row (or none), then launch straight into the auto-rotating infinite-
wing view. Writes gpu/fluidx3d/tunnel_run.txt (the exe's fixed config path)
and starts FluidX3D.exe -- the same mechanism tunnel.bat uses, with knobs.

Design: dark, minimal, grouped controls with live value readouts (layout
follows Apple HIG grouping/spacing conventions translated to ttk: one task
per group, primary action prominent and isolated at the bottom).

Run:  python gpu/fluidx3d/launcher.py     (or double-click launcher.bat)
"""
from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXE = Path(r"L:\Dev\FluidX3D\bin\FluidX3D.exe")
EXE_DIR = Path(r"L:\Dev\FluidX3D")
RUN_CFG = HERE / "tunnel_run.txt"
# All articles are vaneless: the VG row and speck are lattice-stamped.
# Filenames resolve from trailing-edge kind + test-section width; the speck
# self-centers (stamped at domain mid-span) for either width.
TE_TOKEN = {
    "Aileron + gap (neutral)": "wing_clean_ail_n",
    "Aileron up 15°":          "wing_clean_ail_u15",
    "Aileron down 15°":        "wing_clean_ail_d15",
    "Smooth (no gap)":         "wing_a0_binary",
}


def stl_for(te_kind: str, span_m: float) -> Path:
    base = TE_TOKEN[te_kind]
    if base == "wing_a0_binary" and abs(span_m - 1.5) < 1e-9:
        return HERE / "assets" / "wing_a0_binary.stl"   # legacy untagged name
    return HERE / "assets" / f"{base}_s{span_m:g}m.stl"

CHORD = 0.9022              # aileron-station chord, m [DXF]
NU_AIR = 1.4607e-5          # ISA sea-level kinematic viscosity
MPH_TO_MPS = 0.44704

# Station constants cache (chord fraction -> (skin_off, le_off)); filled
# lazily from the airfoil so moving the row needs no helper scripts.
_station_cache: dict[float, tuple[float, float]] = {}


def station_constants(x_frac: float) -> tuple[float, float]:
    """(vg_skin_off_frac, vg_le_off_frac) for a row at x_frac chord."""
    key = round(x_frac, 4)
    if key not in _station_cache:
        import numpy as np
        from geometry.airfoil import load_airfoil, resample_airfoil
        coords = resample_airfoil(
            load_airfoil(REPO / "geometry" / "ls413.dat"), 241, "blunt")
        le = int(np.argmin(coords[:, 0]))
        upper = coords[:le + 1][::-1]
        y_surf = float(np.interp(x_frac, upper[:, 0], upper[:, 1]))
        y_c = 0.5 * (coords[:, 1].min() + coords[:, 1].max())
        x_c = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        _station_cache[key] = (y_surf - y_c, x_frac - x_c)
    return _station_cache[key]


class Launcher(tk.Tk):
    BG, CARD, FG, SUB = "#101218", "#181c26", "#e8eaf0", "#9aa1b2"
    ACCENT = "#27c93f"

    def __init__(self) -> None:
        super().__init__()
        self.title("Glasair GPU Tunnel")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        self._style()
        self._build()

    # ---------------------------------------------------------------- style
    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=self.BG, foreground=self.FG,
                    fieldbackground=self.CARD, font=("Segoe UI", 10))
        s.configure("Card.TLabelframe", background=self.CARD, borderwidth=0)
        s.configure("Card.TLabelframe.Label", background=self.CARD,
                    foreground=self.SUB, font=("Segoe UI", 9, "bold"))
        s.configure("Card.TFrame", background=self.CARD)
        s.configure("Card.TLabel", background=self.CARD, foreground=self.FG)
        s.configure("Sub.TLabel", background=self.CARD, foreground=self.SUB)
        s.configure("Big.TLabel", background=self.CARD, foreground=self.FG,
                    font=("Segoe UI", 16, "bold"))
        s.configure("Launch.TButton", font=("Segoe UI", 12, "bold"), padding=10)
        s.configure("TCheckbutton", background=self.CARD, foreground=self.FG)
        s.map("TCheckbutton", background=[("active", self.CARD)])
        # Combobox readability: the entry field AND its popup listbox both
        # need explicit dark colors (the popup is a plain Tk listbox that
        # ignores ttk styles -- the white-on-white bug).
        s.configure("TCombobox", foreground=self.FG, arrowcolor=self.FG)
        s.map("TCombobox",
              fieldbackground=[("readonly", self.CARD)],
              foreground=[("readonly", self.FG)],
              selectbackground=[("readonly", self.CARD)],
              selectforeground=[("readonly", self.FG)])
        self.option_add("*TCombobox*Listbox.background", self.CARD)
        self.option_add("*TCombobox*Listbox.foreground", self.FG)
        self.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", self.BG)

    # ---------------------------------------------------------------- build
    def _build(self) -> None:
        pad = {"padx": 14, "pady": (10, 4)}

        # ---- flight conditions card -----------------------------------
        f = ttk.Labelframe(self, text="  FLIGHT CONDITION  ",
                           style="Card.TLabelframe")
        f.pack(fill="x", **pad)
        self.aoa = tk.DoubleVar(value=14.0)
        self.mph = tk.DoubleVar(value=80.0)
        self.aoa_lbl = ttk.Label(f, style="Big.TLabel")
        self.mph_lbl = ttk.Label(f, style="Big.TLabel")
        self._slider(f, "Angle of attack", self.aoa, 0, 22, self.aoa_lbl,
                     lambda v: f"{v:.0f}°")
        self._slider(f, "Airspeed", self.mph, 50, 250, self.mph_lbl,
                     lambda v: f"{v:.0f} mph")

        # ---- VG card ----------------------------------------------------
        g = ttk.Labelframe(self, text="  VORTEX GENERATORS  ",
                           style="Card.TLabelframe")
        g.pack(fill="x", **pad)
        self.vg_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(g, text="Install VG row", variable=self.vg_on
                        ).grid(row=0, column=0, columnspan=2, sticky="w",
                               padx=12, pady=(8, 4))
        self.vg_h = self._combo(g, 1, "Height (mm)", ["6", "8", "10", "12", "16"], "10")
        self.vg_p = self._combo(g, 2, "Spacing (mm)", ["50", "70", "90"], "50")
        self.vg_b = self._combo(g, 3, "Vane angle (deg)", ["12", "15", "18"], "15")
        self.vg_x = self._combo(g, 4, "Row at chord %", ["5", "7", "10"], "7")
        self.vg_t = self._combo(g, 5, "Thickness (mm)", ["1.5 (physical)", "6 (visual)"], "6 (visual)")

        # ---- airframe card ----------------------------------------------
        w = ttk.Labelframe(self, text="  AIRFRAME  ", style="Card.TLabelframe")
        w.pack(fill="x", padx=14, pady=(10, 4))
        self.te_kind = self._combo(
            w, 0, "Trailing edge",
            ["Aileron + gap (neutral)", "Aileron up 15°",
             "Aileron down 15°", "Smooth (no gap)"],
            "Aileron + gap (neutral)")
        # Lattice fidelity: finer cells smooth the voxel staircase on the
        # wing skin (spanwise-aligned steps seed artificial roller coherence)
        # at the cost of frame rate. VRAM budget maps to ~cell size.
        self.res = self._combo(
            w, 2, "Resolution",
            ["Fast (6 GB, ~3.6 mm)", "Balanced (12 GB, ~2.9 mm)",
             "Fine (24 GB, ~2.3 mm)"],
            "Balanced (12 GB, ~2.9 mm)")
        # Half-width section: fewer cells at the same cell size = faster
        # stepping. Trade-off: 0.75 m is ~0.8 chords of span, so the largest
        # 3D stall structures get less room than the full 1.66-chord arena.
        self.width = self._combo(
            w, 3, "Section width",
            ["Full (1.5 m arena)", "Half (0.75 m, faster)"],
            "Full (1.5 m arena)")
        self.speck_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(w, text="Speck (breaks fake 2D rollers — honest air)",
                        variable=self.speck_on
                        ).grid(row=1, column=0, columnspan=2, sticky="w",
                               padx=12, pady=(4, 10))

        # ---- action -----------------------------------------------------
        a = ttk.Frame(self, style="Card.TFrame")
        a.pack(fill="x", padx=14, pady=(10, 4))
        ttk.Button(a, text="LAUNCH  ➤", style="Launch.TButton",
                   command=self.launch).pack(fill="x", padx=10, pady=(10, 4))
        ttk.Button(a, text="Stop simulation", command=self.stop
                   ).pack(fill="x", padx=10, pady=(0, 10))
        self.status = ttk.Label(self, text="ready", style="Sub.TLabel",
                                background=self.BG, foreground=self.SUB)
        self.status.pack(fill="x", padx=16, pady=(2, 10))

    def _slider(self, parent, label, var, lo, hi, value_lbl, fmt) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", padx=12, pady=4)
        ttk.Label(row, text=label, style="Sub.TLabel").pack(anchor="w")
        inner = ttk.Frame(row, style="Card.TFrame")
        inner.pack(fill="x")
        scale = ttk.Scale(inner, from_=lo, to=hi, variable=var,
                          command=lambda _v: value_lbl.config(text=fmt(var.get())))
        scale.pack(side="left", fill="x", expand=True, pady=2)
        value_lbl.config(text=fmt(var.get()))
        value_lbl.pack(side="right", padx=(10, 0))

    def _combo(self, parent, row, label, values, default) -> tk.StringVar:
        ttk.Label(parent, text=label, style="Sub.TLabel"
                  ).grid(row=row, column=0, sticky="w", padx=12, pady=3)
        var = tk.StringVar(value=default)
        ttk.Combobox(parent, textvariable=var, values=values, width=14,
                     state="readonly").grid(row=row, column=1, sticky="e",
                                            padx=12, pady=3)
        return var

    # ---------------------------------------------------------------- run
    def launch(self) -> None:
        aoa = self.aoa.get()
        mph = self.mph.get()
        re_value = mph * MPH_TO_MPS * CHORD / NU_AIR
        x_frac = float(self.vg_x.get()) / 100.0
        skin_off, le_off = station_constants(x_frac)
        vg_t = self.vg_t.get().split()[0]
        span_m = 1.5 if self.width.get().startswith("Full") else 0.75
        stl = stl_for(self.te_kind.get(), span_m)
        cfg = (
            "# written by launcher.py\n"
            f"mode        = slice\n"
            f"stl         = {stl.as_posix()}\n"
            f"aoa_deg     = {aoa:.1f}\n"
            f"re          = {re_value:.4g}\n"
            f"vram_mb     = {int(self.res.get().split('(')[1].split()[0])*1000}\n"
            f"u           = 0.075\n"          # stability protocol: safe at every angle
            f"span_m      = {span_m:g}\n"
            f"chord_m     = {CHORD}\n"
            f"autorotate  = 1\n"
            f"t_end_steps = 0\n"
            f"t_end_si    = 0\n"
            f"video_s     = 0\n"
            f"log_every   = 2000\n"
            f"csv         = {(HERE / 'results' / 'forces_live.csv').as_posix()}\n"
            f"vg_enable   = {1 if self.vg_on.get() else 0}\n"
            f"vg_h_mm     = {self.vg_h.get()}\n"
            f"vg_pitch_mm = {self.vg_p.get()}\n"
            f"vg_beta_deg = {self.vg_b.get()}\n"
            f"vg_t_mm     = {vg_t}\n"
            f"vg_skin_off_frac = {skin_off:.5f}\n"
            f"vg_le_off_frac   = {le_off:.5f}\n"
            f"speck_enable = {1 if self.speck_on.get() else 0}\n")
        RUN_CFG.write_text(cfg, encoding="utf-8")
        self.stop(quiet=True)
        subprocess.Popen([str(EXE)], cwd=str(EXE_DIR))
        vg_txt = (f"VG {self.vg_h.get()}mm/{self.vg_p.get()}mm at "
                  f"{self.vg_x.get()}%c" if self.vg_on.get() else "no VGs")
        self.status.config(
            text=f"flying: {aoa:.0f} deg, {mph:.0f} mph (Re {re_value:.2e}), {vg_txt}")

    def stop(self, quiet: bool = False) -> None:
        subprocess.run(["taskkill", "/IM", "FluidX3D.exe", "/F"],
                       capture_output=True)
        if not quiet:
            self.status.config(text="simulation stopped")


if __name__ == "__main__":
    Launcher().mainloop()
