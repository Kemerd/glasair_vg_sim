# -*- coding: utf-8 -*-
"""
Glasair VG Tunnel launcher GUI: pick a geometry (wing section, wing slice,
or the full tail assembly), dial in analytic VG rows and flow conditions,
and launch FluidX3D -- no hand-editing of tunnel_config.txt needed.

Design notes (kept deliberately simple and fast):
  * VGs are NOT baked into STLs from here: the tunnel stamps vanes into the
    lattice analytically (setup_glasair.cpp), so every VG change is a pure
    config change. The GUI just writes the vg_* / vg_elev_* / vg_rud_* keys.
  * Tail articles are generated on demand (make_tail_assembly.py --single)
    in a worker thread the first time a deflection combo is requested, so
    the UI never freezes on a 60 s geometry build.
  * tunnel_run.txt is the per-run config contract shared with tunnel.bat
    and the suite runners; this GUI writes the same format.

Run:  python gpu/fluidx3d/tunnel_gui.py   (or double-click tunnel_gui.bat)
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
RUN_CFG = HERE / "tunnel_run.txt"
RESULTS = HERE / "results"
EXE = Path("L:/Dev/FluidX3D/bin/FluidX3D.exe")
EXE_DIR = Path("L:/Dev/FluidX3D")

NU_AIR = 1.4607e-5            # ISA sea-level kinematic viscosity [m^2/s]
WING_CHORD = 0.9022           # aileron-station chord [DXF]
TAIL_CHORD = 0.70048          # stab root chord [DXF] -- tail Re reference

# ---- palette: dark, minimal, one accent --------------------------------------
BG = "#16181d"
CARD = "#1f2229"
FG = "#e9eaee"
DIM = "#9a9fab"
ACCENT = "#4f9cf7"
ACCENT_DK = "#3c80d0"


def fmt(v: float) -> str:
    """Trim trailing zeros for config values (15.0 -> '15')."""
    return f"{v:g}"


class TunnelGUI:
    """Single-window launcher; everything lives in three cards + footer."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.msgq: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self._style()
        self._build()
        self._refresh()
        self.root.after(150, self._poll)

    # ------------------------------------------------------------------ style
    def _style(self) -> None:
        self.root.title("Glasair VG Tunnel")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Card.TFrame", background=CARD)
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("Dim.TLabel", background=CARD, foreground=DIM,
                    font=("Segoe UI", 9))
        s.configure("Head.TLabel", background=CARD, foreground=DIM,
                    font=("Segoe UI", 9, "bold"))
        s.configure("Title.TLabel", background=BG, foreground=FG,
                    font=("Segoe UI Semibold", 16))
        s.configure("Sub.TLabel", background=BG, foreground=DIM,
                    font=("Segoe UI", 9))
        s.configure("Card.TRadiobutton", background=CARD, foreground=FG)
        s.map("Card.TRadiobutton", background=[("active", CARD)])
        s.configure("Card.TCheckbutton", background=CARD, foreground=FG)
        s.map("Card.TCheckbutton", background=[("active", CARD)])
        s.configure("TEntry", fieldbackground="#2a2e37", foreground=FG,
                    insertcolor=FG, bordercolor=CARD)
        s.configure("TCombobox", fieldbackground="#2a2e37", foreground=FG,
                    background="#2a2e37", arrowcolor=DIM)
        s.configure("Launch.TButton", font=("Segoe UI Semibold", 11),
                    background=ACCENT, foreground="#ffffff", padding=(18, 9))
        s.map("Launch.TButton", background=[("active", ACCENT_DK),
                                            ("disabled", "#33373f")])

    # ------------------------------------------------------------------ build
    def _card(self, parent: tk.Widget, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame")
        outer.pack(fill="x", padx=16, pady=(0, 10))
        ttk.Label(outer, text=title.upper(), style="Head.TLabel"
                  ).pack(anchor="w", padx=14, pady=(10, 2))
        inner = ttk.Frame(outer, style="Card.TFrame")
        inner.pack(fill="x", padx=14, pady=(0, 12))
        return inner

    def _build(self) -> None:
        ttk.Label(self.root, text="Glasair VG Tunnel", style="Title.TLabel"
                  ).pack(anchor="w", padx=16, pady=(14, 0))
        ttk.Label(self.root, text="GPU wind tunnel launcher — geometry, "
                  "VGs and airspeed without config editing",
                  style="Sub.TLabel").pack(anchor="w", padx=16, pady=(0, 12))

        # ---- geometry card ---------------------------------------------------
        g = self._card(self.root, "Geometry")
        self.geom = tk.StringVar(value="tail")
        for col, (val, text) in enumerate((
                ("wing", "Wing — 1.5 m section"),
                ("slice", "Wing — 0.25 m slice"),
                ("tail", "Full tail assembly"))):
            ttk.Radiobutton(g, text=text, value=val, variable=self.geom,
                            style="Card.TRadiobutton", command=self._refresh
                            ).grid(row=0, column=col, sticky="w", padx=(0, 16))
        # wing sub-option: aileron article
        self.ail = tk.StringVar(value="No aileron split")
        self.row_wing = ttk.Frame(g, style="Card.TFrame")
        ttk.Label(self.row_wing, text="Aileron", style="Card.TLabel"
                  ).pack(side="left", padx=(0, 8))
        ttk.Combobox(self.row_wing, textvariable=self.ail, width=22,
                     state="readonly",
                     values=("No aileron split", "Neutral (gap open)",
                             "Down 15°", "Up 15°")
                     ).pack(side="left")
        # tail sub-options: control deflections
        self.elev = tk.StringVar(value="Neutral")
        self.rud = tk.StringVar(value="Neutral")
        self.row_tail = ttk.Frame(g, style="Card.TFrame")
        ttk.Label(self.row_tail, text="Elevator", style="Card.TLabel"
                  ).pack(side="left", padx=(0, 8))
        ttk.Combobox(self.row_tail, textvariable=self.elev, width=18,
                     state="readonly",
                     values=("Neutral", "Up 15° (nose-up)",
                             "Down 15°")).pack(side="left", padx=(0, 18))
        ttk.Label(self.row_tail, text="Rudder", style="Card.TLabel"
                  ).pack(side="left", padx=(0, 8))
        ttk.Combobox(self.row_tail, textvariable=self.rud, width=14,
                     state="readonly", values=("Neutral", "Deflected 15°")
                     ).pack(side="left")

        # ---- VG card ----------------------------------------------------------
        v = self._card(self.root, "Vortex generators — stamped at launch, no STL rebake")
        self.vg_wing = tk.BooleanVar(value=False)
        self.vg_elev = tk.BooleanVar(value=False)
        self.vg_rud = tk.BooleanVar(value=False)
        self.row_vg_wing = ttk.Checkbutton(
            v, text="Wing row — 7% chord, upper surface",
            variable=self.vg_wing, style="Card.TCheckbutton")
        self.row_vg_elev = ttk.Checkbutton(
            v, text="Elevator row — stab underside, 100 mm ahead of hinge",
            variable=self.vg_elev, style="Card.TCheckbutton")
        self.row_vg_rud = ttk.Checkbutton(
            v, text="Rudder rows — both fin sides (analog, no flight-test source)",
            variable=self.vg_rud, style="Card.TCheckbutton")
        # Junction cluster: the owner-sketched corner experiment (see
        # results/concept.jpg and concept2.jpg) with two candidate seatings.
        self.vg_junc = tk.BooleanVar(value=False)
        self.junc_loc = tk.StringVar(value="Slab diagonal (concept 1)")
        self.row_vg_junc = ttk.Frame(v, style="Card.TFrame")
        ttk.Checkbutton(self.row_vg_junc,
                        text="Junction cluster — fin/stab corner, 2 pairs/side at",
                        variable=self.vg_junc, style="Card.TCheckbutton"
                        ).pack(side="left")
        ttk.Combobox(self.row_vg_junc, textvariable=self.junc_loc, width=26,
                     state="readonly",
                     values=("Slab diagonal (concept 1)",
                             "Hinge hugger (concept 2)")
                     ).pack(side="left", padx=(6, 0))
        nums = ttk.Frame(v, style="Card.TFrame")
        self.vg_h = tk.StringVar(value="10")
        self.vg_p = tk.StringVar(value="30")
        self.vg_t = tk.StringVar(value="6")
        for label, var, hint in (("Height mm", self.vg_h, ""),
                                 ("Pair pitch mm", self.vg_p, ""),
                                 ("Thickness mm", self.vg_t,
                                  "≥ 1.5× cell on coarse runs")):
            ttk.Label(nums, text=label, style="Card.TLabel"
                      ).pack(side="left", padx=(0, 6))
            ttk.Entry(nums, textvariable=var, width=6).pack(side="left",
                                                            padx=(0, 4))
            if hint:
                ttk.Label(nums, text=hint, style="Dim.TLabel"
                          ).pack(side="left", padx=(0, 10))
            else:
                ttk.Label(nums, text=" ", style="Dim.TLabel"
                          ).pack(side="left", padx=(0, 10))
        self.vg_rows = (self.row_vg_wing, self.row_vg_elev, self.row_vg_rud)
        self.vg_nums = nums

        # ---- flow card ---------------------------------------------------------
        f = self._card(self.root, "Flow")
        self.mph = tk.StringVar(value="70")
        self.aoa = tk.StringVar(value="8")
        self.vram = tk.StringVar(value="12000")
        self.u_lat = tk.StringVar(value="0.10")
        self.autorot = tk.BooleanVar(value=True)
        for label, var, w in (("Airspeed mph", self.mph, 6),
                              ("AoA °", self.aoa, 5),
                              ("VRAM MB", self.vram, 7),
                              ("Lattice u", self.u_lat, 6)):
            ttk.Label(f, text=label, style="Card.TLabel"
                      ).pack(side="left", padx=(0, 6))
            e = ttk.Entry(f, textvariable=var, width=w)
            e.pack(side="left", padx=(0, 14))
            e.bind("<KeyRelease>", lambda _e: self._update_re())
        self.re_lbl = ttk.Label(f, text="", style="Dim.TLabel")
        self.re_lbl.pack(side="left")

        # ---- footer -------------------------------------------------------------
        foot = ttk.Frame(self.root)
        foot.pack(fill="x", padx=16, pady=(2, 14))
        foot.configure(style="TFrame")
        ttk.Checkbutton(foot, text="Autorotate camera", variable=self.autorot
                        ).pack(side="left")
        self.launch_btn = ttk.Button(foot, text="Launch Tunnel",
                                     style="Launch.TButton",
                                     command=self._launch)
        self.launch_btn.pack(side="right")
        self.status = ttk.Label(self.root, text="Ready.", style="Sub.TLabel")
        self.status.pack(anchor="w", padx=16, pady=(0, 10))

    # ------------------------------------------------------------ view logic
    def _refresh(self) -> None:
        """Show only the sub-rows that apply to the selected geometry."""
        tail = self.geom.get() == "tail"
        self.row_wing.grid_forget()
        self.row_tail.grid_forget()
        if tail:
            self.row_tail.grid(row=1, column=0, columnspan=3, sticky="w",
                               pady=(8, 0))
        else:
            self.row_wing.grid(row=1, column=0, columnspan=3, sticky="w",
                               pady=(8, 0))
        for r in self.vg_rows:
            r.pack_forget()
        self.vg_nums.pack_forget()
        self.row_vg_junc.pack_forget()
        if tail:
            self.row_vg_elev.pack(anchor="w")
            self.row_vg_rud.pack(anchor="w")
            self.row_vg_junc.pack(anchor="w")
            # 1.5 mm physical plates: the lattice stamper enforces a one-cell
            # minimum footprint, so thin plates stay visible on coarse runs.
            if self.vg_t.get() == "6":
                self.vg_t.set("1.5")
        else:
            self.row_vg_wing.pack(anchor="w")
            if self.vg_t.get() == "1.5":
                self.vg_t.set("6")
            if self.vg_p.get() == "30":
                self.vg_p.set("50")
        self.vg_nums.pack(anchor="w", pady=(6, 0))
        self._update_re()

    def _chord(self) -> float:
        return TAIL_CHORD if self.geom.get() == "tail" else WING_CHORD

    def _update_re(self) -> None:
        try:
            re_val = float(self.mph.get()) * 0.44704 * self._chord() / NU_AIR
            self.re_lbl.config(
                text=f"Re = {re_val:.3g}  @ {self._chord():.3f} m chord")
        except ValueError:
            self.re_lbl.config(text="Re = —")

    # ------------------------------------------------------------ article paths
    def _tail_defl(self) -> tuple[float, float]:
        e = {"Neutral": 0.0, "Up 15° (nose-up)": -15.0,
             "Down 15°": +15.0}[self.elev.get()]
        r = {"Neutral": 0.0, "Deflected 15°": +15.0}[self.rud.get()]
        return e, r

    def _stl(self) -> Path:
        g = self.geom.get()
        if g == "tail":
            # Same naming contract as make_tail_assembly.single_tag().
            sys.path.insert(0, str(HERE))
            from make_tail_assembly import single_tag
            e, r = self._tail_defl()
            return ASSETS / f"{single_tag(e, r)}.stl"
        span = "_s0.25m" if g == "slice" else ""
        name = {"No aileron split": f"wing_a0_binary{span}.stl",
                "Neutral (gap open)": f"wing_clean_ail_n_s{0.25 if g == 'slice' else 1.5:g}m.stl",
                "Down 15°": f"wing_clean_ail_d15_s{0.25 if g == 'slice' else 1.5:g}m.stl",
                "Up 15°": f"wing_clean_ail_u15_s{0.25 if g == 'slice' else 1.5:g}m.stl",
                }[self.ail.get()]
        return ASSETS / name

    # ------------------------------------------------------------ config write
    def _write_config(self, stl: Path) -> None:
        g = self.geom.get()
        chord = self._chord()
        re_val = float(self.mph.get()) * 0.44704 * chord / NU_AIR
        lines = [
            "# auto-generated by tunnel_gui.py",
            f"mode        = {'slice' if g == 'slice' else 'wing'}",
            f"stl         = {stl.as_posix()}",
            f"aoa_deg     = {float(self.aoa.get()):g}",
            f"re          = {re_val:.4g}",
            f"vram_mb     = {float(self.vram.get()):g}",
            f"u           = {float(self.u_lat.get()):g}",
            f"chord_m     = {chord}",
            f"autorotate  = {1 if self.autorot.get() else 0}",
            "t_end_steps = 0",
            f"csv         = {(RESULTS / 'forces_live.csv').as_posix()}",
            # Anti-vanish floor for stamped vanes: crisp plates on fine
            # lattices, fused-but-visible plates on coarse ones.
            f"vg_floor_cells = {0.5 if float(self.vram.get()) >= 10000 else 0.75}",
        ]
        if g == "tail":
            # True span + the STL-origin -> bbox-center anchors the analytic
            # tail stamper needs (exact per article: deflections shift the
            # bbox a few millimeters, so measure rather than assume).
            import trimesh
            b = trimesh.load(stl).bounds
            cx, cy = (b[0][0] + b[1][0]) / 2.0, (b[0][1] + b[1][1]) / 2.0
            # span_m must be the LONGEST bbox side: read_stl anchors the
            # lattice scale on it, so if a future article ever grows longer
            # than its span the chord scaling still comes out right.
            span = float(max(b[1] - b[0]))
            lines += [
                f"span_m      = {span:.4f}",
                f"tail_origin_x_mm = {-cx * 1000.0:.1f}",
                f"tail_origin_z_mm = {-cy * 1000.0:.1f}",
                f"vg_elev_enable = {1 if self.vg_elev.get() else 0}",
                f"vg_rud_enable  = {1 if self.vg_rud.get() else 0}",
                f"vg_elev_h_mm     = {fmt(float(self.vg_h.get()))}",
                f"vg_rud_h_mm      = {fmt(float(self.vg_h.get()))}",
                f"vg_elev_pitch_mm = {fmt(float(self.vg_p.get()))}",
                f"vg_rud_pitch_mm  = {fmt(float(self.vg_p.get()))}",
                f"vg_elev_t_mm     = {fmt(float(self.vg_t.get()))}",
                f"vg_rud_t_mm      = {fmt(float(self.vg_t.get()))}",
                f"vg_junc_enable = {1 if self.vg_junc.get() else 0}",
                f"vg_junc_loc    = {2 if 'concept 2' in self.junc_loc.get() else 1}",
                f"vg_junc_t_mm   = {fmt(float(self.vg_t.get()))}",
            ]
        else:
            lines += [
                f"span_m      = {0.25 if g == 'slice' else 1.5}",
                f"vg_enable   = {1 if self.vg_wing.get() else 0}",
                f"vg_h_mm     = {fmt(float(self.vg_h.get()))}",
                f"vg_pitch_mm = {fmt(float(self.vg_p.get()))}",
                f"vg_t_mm     = {fmt(float(self.vg_t.get()))}",
            ]
        RUN_CFG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ launch path
    def _launch(self) -> None:
        try:
            stl = self._stl()
        except Exception as exc:                     # bad entry fields etc.
            self.status.config(text=f"Error: {exc}")
            return
        if stl.exists():
            self._go(stl)
            return
        if self.geom.get() != "tail":
            self.status.config(text=f"Missing asset: {stl.name} — run "
                               "make_vg_wing.py / make_aileron_wing.py first.")
            return
        # Tail combo not baked yet: generate it in a worker thread so the
        # window stays alive, then launch from the poll loop when done.
        e, r = self._tail_defl()
        self.launch_btn.state(["disabled"])
        self.status.config(text=f"Generating {stl.name} (about a minute)...")

        def work() -> None:
            proc = subprocess.run(
                [sys.executable, str(HERE / "make_tail_assembly.py"),
                 "--single", fmt(e), fmt(r), "--no-render"],
                capture_output=True, text=True)
            self.msgq.put(("done" if proc.returncode == 0 else "fail",
                           stl.as_posix() if proc.returncode == 0
                           else (proc.stderr or "generation failed")[-300:]))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _go(self, stl: Path) -> None:
        self._write_config(stl)
        RESULTS.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([str(EXE)], cwd=str(EXE_DIR))
        vg_bits = []
        if self.geom.get() == "tail":
            if self.vg_elev.get():
                vg_bits.append("elevator VG")
            if self.vg_rud.get():
                vg_bits.append("rudder VG")
            if self.vg_junc.get():
                vg_bits.append("junction VG "
                               + ("loc2" if "concept 2" in self.junc_loc.get()
                                  else "loc1"))
        elif self.vg_wing.get():
            vg_bits.append("wing VG")
        self.status.config(text=f"Launched: {stl.name}"
                           + (f"  +  {' + '.join(vg_bits)}" if vg_bits else "")
                           + f"  @  {self.mph.get()} mph, {self.aoa.get()}° AoA")

    def _poll(self) -> None:
        try:
            kind, payload = self.msgq.get_nowait()
        except queue.Empty:
            pass
        else:
            self.launch_btn.state(["!disabled"])
            if kind == "done":
                self._go(Path(payload))
            else:
                self.status.config(text=f"Generation failed: {payload}")
        self.root.after(150, self._poll)


def main() -> None:
    root = tk.Tk()
    TunnelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
