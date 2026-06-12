# -*- coding: utf-8 -*-
"""
RapidCFD (GPU OpenFOAM 2.3 fork) case factory for the Glasair wing VG study.

WHAT THIS BUILDS
  Span-periodic slice cases of the aileron-station wing section with NO
  aileron gap (the clean single-solid article from geometry/stl_gen.py,
  include_gap=False equivalent) plus variants wearing one counter-rotating
  vortex-generator pair, meshed with stock OpenFOAM v2506 snappyHexMesh on
  the CPU and solved by RapidCFD's simpleFoam on the RTX 5090.

DOMAIN STRATEGY (one VG pitch, mirror walls)
  A counter-rotating VG array with pair pitch P is mirror-symmetric about
  the planes midway between pairs AND about each pair's own centerline, so
  a slice exactly one pitch wide with symmetry (slip) side walls reproduces
  the infinite array EXACTLY for the steady mean flow - no cyclic/AMI
  machinery, which keeps every boundary type on RapidCFD's well-trodden
  GPU path. The vane pair sits centered: vanes at z = +/- P/4, toe-out.

  The wing solid is extruded 20% wider than the domain so it pierces the
  side planes cleanly (snappy clips it; no degenerate slivers at the
  corners). Angle of attack is baked into the STL (geometry rotated about
  the quarter chord) so the far-field box stays axis-aligned and the
  force decomposition is simply lift = +y, drag = +x.

MESH FAMILY PARITY
  Clean and VG cases share the identical background mesh, boxes, surface
  levels and layer recipe; the only difference is the vane-distance
  refinement band (meaningless on a vaneless article). Case-to-case deltas
  therefore ride on near-identical mesh families.

PRECISION NOTE
  RapidCFD is built WM_PRECISION_OPTION=SP (fp32 fields - full bandwidth
  on the GeForce, whose native fp64 runs at 1/64 rate) with the in-house
  patch that emulates fp64 where it actually matters: every solver-side
  global reduction (CG dot products, residual norms, norm factors)
  accumulates in double on the GPU. See docs/BACKLOG.md and the patched
  RapidCFD tree in WSL (~/RapidCFD-dev).

OUTPUTS
  gpu/rapidcfd/assets/<article>.stl        rotated wall + vane solids
  gpu/rapidcfd/assets/<article>_vanes.stl  vanes-only (refinement source)
  gpu/rapidcfd/cases/<case>/               complete OpenFOAM case dirs
  gpu/rapidcfd/run_all.sh                  WSL-side mesh+solve driver

Run:  python gpu/rapidcfd/build_cases.py
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import trimesh

from geometry.airfoil import load_airfoil, resample_airfoil
from geometry.stl_gen import extrude_section
from geometry.units import load_aircraft

HERE = REPO / "gpu" / "rapidcfd"
ASSETS = HERE / "assets"
CASES = HERE / "cases"

# ---------------------------------------------------------------------------
#  Flow & study constants
# ---------------------------------------------------------------------------
NU = 1.48e-05            # m^2/s, standard air (matches the M1 pipeline)
RE = 2.2e6               # stall-regime chord Reynolds (80 mph @ 0.9022 m)
SPAN_OVERHANG = 1.2      # STL span / domain span: pierce the side planes

# The study cases: (name, vane height mm or None, alpha deg, vane x/c or None)
# - clean_a08 is the pipeline-sanity point (attached flow, Cl well known
#   from XFOIL/NASA - fully-turbulent kOmegaSST should land ~5-10% low)
# - a18 is the discriminating angle from the FluidX3D Act-III post-mortem
# - x/c None = the IMP74 default station (chord_position_frac, 0.07)
# - the xNN cases sweep the row aft: fielded Glasair installs have been
#   spotted anywhere from just behind the LE to roughly mid-chord, so the
#   sweep brackets that range at the 12 mm height / 50 mm pitch point
CASE_MATRIX = [
    ("clean_a08", None, 8.0, None),
    ("clean_a18", None, 18.0, None),
    ("vg12p50_a18", 12.0, 18.0, None),
    ("vg16p50_a18", 16.0, 18.0, None),
    ("vg12x15_a18", 12.0, 18.0, 0.15),
    ("vg12x30_a18", 12.0, 18.0, 0.30),
    ("vg12x45_a18", 12.0, 18.0, 0.45),
]


# ---------------------------------------------------------------------------
#  Geometry: clean / VG'd slice articles, alpha baked in
# ---------------------------------------------------------------------------

def upper_surface_point(coords: np.ndarray, x_frac: float) -> tuple[float, float]:
    """(y/c, local slope) of the upper surface at chordwise station x_frac.

    Same construction as gpu/fluidx3d/make_vg_wing.py: the resampled loop is
    Selig-ordered (TE-upper -> LE -> TE-lower) so the upper surface is the
    first half, reversed to ascending x for interpolation.
    """
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]
    y = float(np.interp(x_frac, upper[:, 0], upper[:, 1]))
    i = int(np.searchsorted(upper[:, 0], x_frac))
    i = max(1, min(i, len(upper) - 1))
    dx, dy = (upper[i] - upper[i - 1])
    return y, float(np.arctan2(dy, dx))


def make_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One rectangular vane plate: LE at origin, +x along the plate, +y up."""
    box = trimesh.creation.box(extents=(length, h, thick))
    box.apply_translation((length / 2.0, h / 2.0, 0.0))
    return box


def build_article(name: str, h_mm: float | None, alpha_deg: float,
                  ac, coords: np.ndarray, domain_span: float,
                  x_frac_override: float | None = None) -> tuple[Path, Path | None]:
    """Build one wall article (wing [+ vane pair]), rotated to alpha.

    Returns (wall_stl, vanes_stl-or-None). The vanes-only STL is exported
    separately because snappy uses it purely as a distance-refinement
    source; the wall solid is the boolean union so inside/outside stays
    unambiguous.
    """
    chord = ac.wing.aileron.chord_at_mid_station          # 0.9022 m [DXF]
    pitch = ac.vg_defaults.wing.spacing_outboard          # 0.050 m  [IMP74]
    # Chordwise station of the vane row: study default from IMP74 unless a
    # matrix entry overrides it (the placement-sweep cases).
    x_frac = (x_frac_override if x_frac_override is not None
              else ac.vg_defaults.wing.chord_position_frac)  # 0.07   [IMP74]
    beta = ac.vg_defaults.vane_incidence                  # 15 deg, radians
    l_per_h = ac.vg_defaults.vane_length_per_height       # 3.0

    stl_span = SPAN_OVERHANG * domain_span

    # Clean gapless wing solid - the "no aileron gap" article. The blunt-TE
    # resampled loop and extrusion come from the validated M0 toolkit.
    wing = extrude_section(coords, chord, stl_span)

    vanes: list[trimesh.Trimesh] = []
    if h_mm is not None:
        h = h_mm / 1000.0
        vane_l = l_per_h * h
        vane_t = max(0.0015, h / 8.0)                     # physical plate
        y_surf, slope = upper_surface_point(coords, x_frac)
        x_le, y_le = x_frac * chord, y_surf * chord

        # One toe-out pair centered on the slice: vanes at z = +/- pitch/4,
        # incidence +/- beta, laid flush on the local skin slope and sunk
        # 0.1h so the union has guaranteed overlap.
        for sgn in (+1.0, -1.0):
            v = make_vane(h, vane_l, vane_t)
            v.apply_transform(trimesh.transformations.rotation_matrix(
                sgn * beta, (0.0, 1.0, 0.0)))
            v.apply_transform(trimesh.transformations.rotation_matrix(
                slope, (0.0, 0.0, 1.0)))
            v.apply_translation((x_le, y_le - 0.10 * h, sgn * pitch / 4.0))
            vanes.append(v)

        wall = trimesh.boolean.union([wing] + vanes)
    else:
        wall = wing

    # Bake the angle of attack into the geometry: nose-up alpha = rotate
    # -alpha about z (x points downstream, y up), about the quarter chord
    # so the moment reference is trivially (c/4, 0, 0) in case coordinates.
    rot = trimesh.transformations.rotation_matrix(
        -math.radians(alpha_deg), (0.0, 0.0, 1.0), (0.25 * chord, 0.0, 0.0))
    wall.apply_transform(rot)

    ASSETS.mkdir(parents=True, exist_ok=True)
    wall_path = ASSETS / f"{name}_wall.stl"
    wall.export(wall_path)
    print(f"[articles] {wall_path.name}: watertight={wall.is_watertight} "
          f"faces={len(wall.faces)} alpha={alpha_deg:g}deg")

    vanes_path = None
    if vanes:
        vblob = trimesh.util.concatenate(vanes)
        vblob.apply_transform(rot)
        vanes_path = ASSETS / f"{name}_vanes.stl"
        vblob.export(vanes_path)
        print(f"[articles] {vanes_path.name}: refinement source, "
              f"faces={len(vblob.faces)}")
    return wall_path, vanes_path


# ---------------------------------------------------------------------------
#  OpenFOAM dictionaries (2.3-compatible syntax throughout, so the same
#  case feeds v2506 meshing utilities AND the RapidCFD 2.3-era solver)
# ---------------------------------------------------------------------------

def foam_header(cls: str, obj: str, location: str = "system") -> str:
    """Standard FoamFile banner; version 2.0 dict format suits both vintages."""
    return f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{location}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


# Domain box, meters. x: -5c .. +10.5c, y: +/-5.5c, z: one VG pitch.
XMIN, XMAX = -4.5, 9.5
YMIN, YMAX = -5.0, 5.0
BASE = 0.1               # background cell size in x,y (z = pitch, 1 cell)


def block_mesh_dict(pitch: float) -> str:
    nx = round((XMAX - XMIN) / BASE)
    ny = round((YMAX - YMIN) / BASE)
    zmin, zmax = -pitch / 2.0, pitch / 2.0
    return foam_header("dictionary", "blockMeshDict") + f"""
// Background hex box for snappy: base {BASE} m cells in x/y and a single
// cell across the one-pitch span (snappy's refinement subdivides z too,
// reaching sub-mm at the vane band).
//
// Side boundaries are CYCLIC: true span periodicity is exact for the
// infinite VG array, and it deliberately avoids RapidCFD's transform
// patches (slip/symmetryPlane) whose 2.3-era PISO/SIMPLE algebra injects
// spurious tangential wall friction on the GPU (proven by the uniform-flow
// reproducer; modern OpenFOAM fixed this class of artifact via
// constrainHbyA). Top/bottom are freestream for the same reason - plain
// inletOutlet machinery, no transform path, and less blockage than slip.
convertToMeters 1;

vertices
(
    ({XMIN} {YMIN} {zmin})  ({XMAX} {YMIN} {zmin})
    ({XMAX} {YMAX} {zmin})  ({XMIN} {YMAX} {zmin})
    ({XMIN} {YMIN} {zmax})  ({XMAX} {YMIN} {zmax})
    ({XMAX} {YMAX} {zmax})  ({XMIN} {YMAX} {zmax})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5)); }}
    top      {{ type patch; faces ((3 7 6 2)); }}
    bottom   {{ type patch; faces ((0 1 5 4)); }}
    // Sides are meshed as PLAIN patches (snappy's layer extrusion aborts
    // when wall geometry pierces a cyclic boundary) and converted to a
    // translational cyclic pair afterwards by createPatch - the geometry
    // is z-extruded, so the two planes castellate/snap/layer identically
    // and the faces match by translation.
    sideLeft {{ type patch; faces ((0 3 2 1)); }}
    sideRight{{ type patch; faces ((4 5 6 7)); }}
);

mergePatchPairs ();
"""


def snappy_dict(has_vanes: bool) -> str:
    vane_geom = """
    vanes
    {
        type triSurfaceMesh;
        file "vanes.stl";
    }""" if has_vanes else ""
    vane_refine = """
        // Sub-mm band hugging the vane plates: L8 ~ 0.39 mm within 1.5 mm,
        // L7 within 4 mm - resolves plate-edge vortex roll-up, the actual
        // VG mechanism the LBM arm could never reach.
        vanes
        {
            mode distance;
            levels ((0.0015 8) (0.004 7));
        }""" if has_vanes else ""
    return foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    wing
    {{
        type triSurfaceMesh;
        file "wing.stl";
    }}{vane_geom}

    // Near-field and wake boxes (axis-aligned because alpha is baked into
    // the rotated STL, never into the domain).
    boxNear {{ type searchableBox; min (-0.5 -0.6 -1); max ( 1.4  0.6 1); }}
    boxWake {{ type searchableBox; min (-0.9 -1.0 -1); max ( 3.6  1.0 1); }}
}}

castellatedMeshControls
{{
    maxLocalCells       4000000;
    maxGlobalCells      12000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
        {{ file "wing.eMesh"; level 6; }}
    );

    refinementSurfaces
    {{
        wing
        {{
            // L5-L6 on the skin: 3.1 down to 1.56 mm in x/y (half that in
            // z, the background cell is flatter); curvature picks L6.
            level (5 6);
            patchInfo {{ type wall; }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        wing
        {{
            // Distance bands off the skin: keeps the boundary-layer
            // neighbourhood at L6 and the near-field at L5.
            mode distance;
            levels ((0.012 6) (0.05 5));
        }}{vane_refine}
        boxNear {{ mode inside; levels ((1e15 4)); }}
        boxWake {{ mode inside; levels ((1e15 3)); }}
    }}

    locationInMesh (-2.0 -1.0 0.0001);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes   false;

    layers
    {{
        // Layers grow on every wing-derived patch (vanes included; snappy
        // collapses them locally where the thin plates pinch).
        "wing.*"
        {{
            nSurfaceLayers 5;
        }}
    }}

    // Absolute sizing: first cell 0.4 mm puts y+ ~ 35-45 at Re 2.2e6 -
    // squarely in the wall-function log-layer window.
    firstLayerThickness 4.0e-4;
    expansionRatio      1.25;
    minThickness        5.0e-5;

    nGrow               0;
    featureAngle        130;
    slipFeatureAngle    30;
    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    // Both spellings: 2.x reads minMedianAxisAngle, v2506 minMedialAxisAngle;
    // each vintage ignores the key it does not know.
    minMedianAxisAngle  90;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;

    relaxed
    {{
        maxNonOrtho 75;
    }}
}}

debug 0;
mergeTolerance 1e-6;
"""


def create_patch_dict(pitch: float) -> str:
    return foam_header("dictionary", "createPatchDict") + f"""
// Post-snappy conversion of the side planes into a translational cyclic
// pair (see blockMeshDict header for why they are meshed as plain patches).
pointSync false;

patches
(
    {{
        name sideL;
        patchInfo
        {{
            type            cyclicAMI;
            neighbourPatch  sideR;
            transform       translational;
            separationVector (0 0 {pitch});
            matchTolerance  1e-3;
        }}
        constructFrom patches;
        patches (sideLeft);
    }}

    {{
        name sideR;
        patchInfo
        {{
            type            cyclicAMI;
            neighbourPatch  sideL;
            transform       translational;
            separationVector (0 0 {-pitch});
            matchTolerance  1e-3;
        }}
        constructFrom patches;
        patches (sideRight);
    }}
);
"""


def surface_features_dict() -> str:
    return foam_header("dictionary", "surfaceFeatureExtractDict") + """
wing.stl
{
    extractionMethod    extractFromSurface;

    extractFromSurfaceCoeffs
    {
        // 150 deg keeps the blunt TE corners and vane plate edges while
        // ignoring the gentle skin curvature.
        includedAngle   150;
    }

    writeObj            no;
}
"""


def control_dict(u_inf: float, chord: float, pitch: float, n_iter: int) -> str:
    aref = chord * pitch
    return foam_header("dictionary", "controlDict") + f"""
application     simpleFoam;

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {n_iter};
deltaT          1;

writeControl    timeStep;
writeInterval   1000;
purgeWrite      2;

writeFormat     ascii;
writePrecision  7;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable false;

functions
{{
    forceCoeffs1
    {{
        // 2.3-era function-object syntax (RapidCFD vintage). Geometry is
        // rotated to alpha, freestream is +x, so lift/drag axes are fixed.
        type            forceCoeffs;
        functionObjectLibs ("libforces.so");
        outputControl   timeStep;
        outputInterval  25;

        patches         (wing);
        pName           p;
        UName           U;
        rhoName         rhoInf;
        rhoInf          1.0;        // kinematic pressure -> rho = 1
        log             true;

        CofR            ({0.25 * chord:.6f} 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);

        magUInf         {u_inf:.4f};
        lRef            {chord:.4f};
        Aref            {aref:.6f};  // chord x one-pitch span
    }}
}}
"""


def fv_schemes(momentum_div: str = "bounded Gauss limitedLinearV 1") -> str:
    return foam_header("dictionary", "fvSchemes") + f"""
ddtSchemes
{{
    default         steadyState;
}}

// RapidCFD's GPU port registers a reduced scheme menu: grad is Gauss-only
// (no cellLimited/leastSquares), and linearUpwind is absent - limitedLinear
// (TVD) is the second-order convection workhorse that survived the port.
gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    // Two-stage convection: the runner solves first on dissipative upwind
    // to kill the impulsive-start global oscillation, then restarts on the
    // TVD scheme for the reported numbers (this file is the stage marker).
    div(phi,U)      {momentum_div};
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear limited 0.33;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         limited 0.33;
}}

fluxRequired
{{
    default         no;
    p;
}}
"""


def fv_solution() -> str:
    return foam_header("dictionary", "fvSolution") + """
solvers
{
    // GPU pressure solve: Krylov + AINV is RapidCFD's massively-parallel
    // sweet spot (DIC-style preconditioners serialize on GPU). With the
    // SP build, the fp64-emulated reductions keep the CG recurrences
    // honest; tolerances sized for fp32 fields.
    p
    {
        solver          PCG;
        preconditioner  AINV;
        tolerance       1e-7;
        relTol          0.01;
        maxIter         1000;
    }

    "(U|k|omega)"
    {
        solver          PBiCGStab;
        preconditioner  AINV;
        tolerance       1e-7;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-5;
        U               1e-6;
        "(k|omega)"     1e-6;
    }
}

// potentialFoam pre-pass settings (initializes U/phi so SIMPLE starts from
// an attached-flow guess instead of the impulsive uniform field whose
// startup vortex was observed to pump a domain-scale Cl limit cycle).
potentialFlow
{
    nNonOrthogonalCorrectors 10;
}

relaxationFactors
{
    // Deliberately heavy damping: the slip-walled slice channel reflects
    // everything, so transients must be killed by relaxation alone.
    fields
    {
        p               0.2;
    }
    equations
    {
        U               0.5;
        k               0.4;
        omega           0.4;
    }
}
"""


def transport_properties() -> str:
    return foam_header("dictionary", "transportProperties", "constant") + f"""
transportModel  Newtonian;

nu              nu [0 2 -1 0 0 0 0] {NU};
"""


def ras_properties() -> str:
    return foam_header("dictionary", "RASProperties", "constant") + """
// kOmegaSST fully-turbulent: RapidCFD carries no transition model (the
// documented limitation that parked it for M1); acceptable here because
// the NASA anchors at Re 2.2e6 are TRIPPED data and the study question is
// the VG-on vs VG-off DELTA, not absolute transition location.
RASModel        kOmegaSST;

turbulence      on;

printCoeffs     on;
"""


def turbulence_properties() -> str:
    return foam_header("dictionary", "turbulenceProperties", "constant") + """
simulationType  RASModel;
"""


def field_file(name: str, dims: str, internal: str, bcs: dict[str, str]) -> str:
    body = foam_header("volScalarField" if name != "U" else "volVectorField",
                       name, "0")
    body += f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for patch, bc in bcs.items():
        body += f"    {patch}\n    {{\n{bc}    }}\n\n"
    body += "}\n"
    return body


def zero_dir(u_inf: float) -> dict[str, str]:
    """All 0/ fields. Freestream turbulence: I=0.5%, nut/nu ~ 8.

    No slip/symmetry anywhere: sides are cyclic (exact span periodicity)
    and top/bottom use the freestream (per-face inletOutlet) family -
    RapidCFD's transform-patch algebra is avoided entirely (GPU artifact,
    see blockMeshDict header).
    """
    k_inf = 1.5 * (u_inf * 0.005) ** 2          # ~0.049 m2/s2
    omega_inf = k_inf / (8.0 * NU)              # nut = k/omega = 8 nu
    cyc = "        type            cyclicAMI;\n"
    fields = {}

    fs_u = (f"        type            freestream;\n"
            f"        freestreamValue uniform ({u_inf:.4f} 0 0);\n")
    fs_k = (f"        type            freestream;\n"
            f"        freestreamValue uniform {k_inf:.6g};\n")
    fs_w = (f"        type            freestream;\n"
            f"        freestreamValue uniform {omega_inf:.6g};\n")

    fields["U"] = field_file(
        "U", "[0 1 -1 0 0 0 0]", f"uniform ({u_inf:.4f} 0 0)",
        {
            "inlet":   f"        type            fixedValue;\n        value           uniform ({u_inf:.4f} 0 0);\n",
            "outlet":  f"        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform ({u_inf:.4f} 0 0);\n",
            "top": fs_u, "bottom": fs_u,
            "sideL": cyc, "sideR": cyc,
            "wing":    "        type            fixedValue;\n        value           uniform (0 0 0);\n",
        })

    fields["p"] = field_file(
        "p", "[0 2 -2 0 0 0 0]", "uniform 0",
        {
            "inlet":   "        type            zeroGradient;\n",
            "outlet":  "        type            fixedValue;\n        value           uniform 0;\n",
            "top": "        type            zeroGradient;\n",
            "bottom": "        type            zeroGradient;\n",
            "sideL": cyc, "sideR": cyc,
            "wing":    "        type            zeroGradient;\n",
        })

    fields["k"] = field_file(
        "k", "[0 2 -2 0 0 0 0]", f"uniform {k_inf:.6g}",
        {
            "inlet":   f"        type            fixedValue;\n        value           uniform {k_inf:.6g};\n",
            "outlet":  f"        type            inletOutlet;\n        inletValue      uniform {k_inf:.6g};\n        value           uniform {k_inf:.6g};\n",
            "top": fs_k, "bottom": fs_k,
            "sideL": cyc, "sideR": cyc,
            "wing":    f"        type            kqRWallFunction;\n        value           uniform {k_inf:.6g};\n",
        })

    fields["omega"] = field_file(
        "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega_inf:.6g}",
        {
            "inlet":   f"        type            fixedValue;\n        value           uniform {omega_inf:.6g};\n",
            "outlet":  f"        type            inletOutlet;\n        inletValue      uniform {omega_inf:.6g};\n        value           uniform {omega_inf:.6g};\n",
            "top": fs_w, "bottom": fs_w,
            "sideL": cyc, "sideR": cyc,
            "wing":    f"        type            omegaWallFunction;\n        value           uniform {omega_inf:.6g};\n",
        })

    fields["nut"] = field_file(
        "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
        {
            "inlet":   "        type            calculated;\n        value           uniform 0;\n",
            "outlet":  "        type            calculated;\n        value           uniform 0;\n",
            "top": "        type            calculated;\n        value           uniform 0;\n",
            "bottom": "        type            calculated;\n        value           uniform 0;\n",
            "sideL": cyc, "sideR": cyc,
            "wing":    "        type            nutkWallFunction;\n        value           uniform 0;\n",
        })
    return fields


# ---------------------------------------------------------------------------
#  Case assembly + the WSL runner
# ---------------------------------------------------------------------------

def write_case(name: str, wall_stl: Path, vanes_stl: Path | None,
               u_inf: float, chord: float, pitch: float, n_iter: int) -> None:
    case = CASES / name
    if case.exists():
        shutil.rmtree(case)
    (case / "system").mkdir(parents=True)
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "0").mkdir()

    # Geometry into the case (snappy reads constant/triSurface/wing.stl).
    shutil.copy2(wall_stl, case / "constant" / "triSurface" / "wing.stl")
    if vanes_stl is not None:
        shutil.copy2(vanes_stl, case / "constant" / "triSurface" / "vanes.stl")

    (case / "system" / "blockMeshDict").write_text(block_mesh_dict(pitch))
    (case / "system" / "snappyHexMeshDict").write_text(snappy_dict(vanes_stl is not None))
    (case / "system" / "createPatchDict").write_text(create_patch_dict(pitch))
    (case / "system" / "surfaceFeatureExtractDict").write_text(surface_features_dict())
    (case / "system" / "controlDict").write_text(control_dict(u_inf, chord, pitch, n_iter))
    # Stage 1 = dissipative upwind (startup damping), stage 2 = TVD scheme
    # for the reported window; the runner swaps the active fvSchemes.
    (case / "system" / "fvSchemes").write_text(
        fv_schemes("bounded Gauss upwind"))
    (case / "system" / "fvSchemes.stage1").write_text(
        fv_schemes("bounded Gauss upwind"))
    (case / "system" / "fvSchemes.stage2").write_text(
        fv_schemes("bounded Gauss limitedLinearV 1"))
    (case / "system" / "fvSolution").write_text(fv_solution())

    (case / "constant" / "transportProperties").write_text(transport_properties())
    (case / "constant" / "RASProperties").write_text(ras_properties())
    (case / "constant" / "turbulenceProperties").write_text(turbulence_properties())

    for fname, content in zero_dir(u_inf).items():
        (case / "0" / fname).write_text(content)

    print(f"[case] {name}: assembled at {case}")


RUNNER = r"""#!/bin/bash
# Mesh (OpenFOAM v2506, CPU) + solve (RapidCFD SP/sm_120, GPU) every case.
# Cases are staged into the WSL filesystem first: the solver writes far too
# often for the 9p-mounted Windows drive.
# (no set -u: the OpenFOAM/RapidCFD bashrc files reference unset vars freely)
set -o pipefail

# MESH_ONLY=1 stops after meshing - used to pre-mesh while RapidCFD compiles.
MESH_ONLY=${MESH_ONLY:-0}
# SKIP_MESH=1 solves an already-staged, already-meshed case in place.
SKIP_MESH=${SKIP_MESH:-0}

SRC="$(dirname "$(readlink -f "$0")")/cases"
STAGE=~/glasair_rapidcfd
RESULTS="$(dirname "$(readlink -f "$0")")/results"
mkdir -p "$STAGE" "$RESULTS"

OF=/usr/lib/openfoam/openfoam2506/etc/bashrc
RC=~/RapidCFD-dev/etc/bashrc

for case in "$@"; do
    src="$SRC/$case"
    dst="$STAGE/$case"

    if [ "$SKIP_MESH" != "1" ]; then
        [ -d "$src" ] || { echo "no such case: $case"; exit 1; }
        rm -rf "$dst"; cp -r "$src" "$dst"

        echo "=== [$case] meshing (v2506) ==="
        (
            source "$OF" || true
            cd "$dst"
            surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 \
                || surfaceFeatures > log.surfaceFeatureExtract 2>&1
            blockMesh           > log.blockMesh 2>&1            || exit 1
            snappyHexMesh -overwrite > log.snappyHexMesh 2>&1   || exit 1
            # Convert the plain side patches into the translational cyclic
            # pair (meshing directly with cyclic sides crashes the layer
            # extrusion when the wall geometry pierces the boundary).
            createPatch -overwrite > log.createPatch 2>&1       || exit 1
            # createPatch may emit the repatched mesh into 0/ depending on
            # startFrom; the solver and renumberMesh expect it in constant/.
            if [ -d 0/polyMesh ]; then
                rm -rf constant/polyMesh
                mv 0/polyMesh constant/polyMesh
            fi
            checkMesh           > log.checkMesh 2>&1
            # No renumberMesh: v2506's renumber rewrites the boundary file
            # before it renumbers fields, and it dies on the 2.3-dialect
            # cyclic field entries - leaving the mesh de-cycled. The GPU
            # solver does not need the bandwidth ordering badly enough to
            # risk that.
        ) || { echo "[$case] MESHING FAILED"
               for f in "$dst"/log.*; do echo "--- $f"; tail -20 "$f"; done
               continue; }
        grep -E "^ *cells:|Mesh OK|Failed" "$dst/log.checkMesh" | head -5
    fi

    [ "$MESH_ONLY" = "1" ] && { echo "=== [$case] mesh-only, skipping solve ==="; continue; }

    echo "=== [$case] solving (RapidCFD, GPU) ==="
    (
        export PATH=/usr/local/cuda/bin:$PATH
        source "$RC"
        cd "$dst"

        # Fresh pseudo-time history: drop any time dirs from earlier runs.
        for d in [0-9]*; do [ "$d" != "0" ] && rm -rf "$d"; done
        rm -rf postProcessing

        # Attached-flow initial guess (writes U/phi into 0/).
        potentialFoam -noFunctionObjects > log.potentialFoam 2>&1

        # Stage 1: first-order momentum, kills the startup transient.
        cp system/fvSchemes.stage1 system/fvSchemes
        sed -i "s/^endTime.*/endTime         ${STAGE1_END:-2000};/" system/controlDict
        simpleFoam > log.simpleFoam.stage1 2>&1

        # Stage 2: restart on the TVD scheme for the reported window.
        cp system/fvSchemes.stage2 system/fvSchemes
        sed -i "s/^endTime.*/endTime         ${STAGE2_END:-4500};/" system/controlDict
        simpleFoam > log.simpleFoam 2>&1
        true
    )
    # RapidCFD exits non-zero from a benign CUDA teardown bug even after a
    # clean run, so success is judged by the solver reaching its End banner.
    if grep -q "^End$" "$dst/log.simpleFoam" 2>/dev/null; then
        tail -5 "$dst/log.simpleFoam"
    else
        echo "[$case] SOLVE FAILED"
        for f in "$dst"/log.potentialFoam "$dst"/log.simpleFoam.stage1 "$dst"/log.simpleFoam; do
            echo "--- $f"; tail -15 "$f" 2>/dev/null
        done
    fi

    # Ship the numbers home (coefficients, logs, last mesh stats).
    out="$RESULTS/$case"
    mkdir -p "$out"
    cp -r "$dst/postProcessing" "$out/" 2>/dev/null
    cp "$dst"/log.* "$out/" 2>/dev/null
    echo "=== [$case] done; results in $out ==="
done
"""


def main() -> None:
    # Optional case filter: `--only nameA nameB` rebuilds just those matrix
    # entries (used to add sweep cases without touching ones already staged
    # or mid-solve in WSL - write_case wipes the case dir it rebuilds).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="CASE",
                    help="build only these CASE_MATRIX entries")
    opts = ap.parse_args()

    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station
    pitch = ac.vg_defaults.wing.spacing_outboard
    u_inf = RE * NU / chord

    print(f"[build] chord={chord:.4f} m  pitch={pitch * 1000:.0f} mm  "
          f"U={u_inf:.2f} m/s (Re {RE:.2g})")

    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")

    selected = [row for row in CASE_MATRIX
                if not opts.only or row[0] in opts.only]
    if opts.only:
        missing = set(opts.only) - {row[0] for row in selected}
        if missing:
            raise SystemExit(f"[build] not in CASE_MATRIX: {sorted(missing)}")

    for name, h_mm, alpha, x_frac in selected:
        wall, vanes = build_article(name, h_mm, alpha, ac, coords, pitch,
                                    x_frac_override=x_frac)
        write_case(name, wall, vanes, u_inf, chord, pitch, n_iter=4000)

    runner = HERE / "run_all.sh"
    runner.write_text(RUNNER, newline="\n")
    print(f"[build] runner: {runner}")
    print("[build] WSL:  bash gpu/rapidcfd/run_all.sh " +
          " ".join(row[0] for row in selected))


if __name__ == "__main__":
    main()
