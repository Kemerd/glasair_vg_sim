"""Unit-tag parsing and aircraft.yaml loading for the Glasair III VG study.

Every dimensional quantity in ``aircraft.yaml`` is written as a mapping
``{value: <num>, unit: "<tag>"}`` so the file stays human-auditable in the
units the source documents use (factory 3-view in feet/inches, Virginia Tech
2005 slides in ft^2 and mph, Strausak's Impulse #74 article in millimeters).
This module is the single choke point where those tags are converted to SI;
no other script in the repository may perform unit conversion.

Conversion factors are the exact legal definitions:
  * 1 in  = 0.0254 m           (international inch, NIST SP 811)
  * 1 ft  = 0.3048 m           (international foot, NIST SP 811)
  * 1 mph = 0.44704 m/s        (derived from the international mile)
  * 1 kt  = 1852/3600 m/s      (international nautical mile, BIPM SI brochure)
  * 1 lb  = 0.45359237 kg      (avoirdupois pound, NIST SP 811)
  * 1 slug = 14.59390294 kg    (lbf s^2/ft, derived from lb and g0)

Angles convert to RADIANS — downstream geometry code (hinge rotations, vane
incidence, taper-alignment corrections) consumes radians exclusively, so the
degree tag is eliminated at load time rather than at every call site.

The public contract consumed by sibling modules (dxf_reader, stl_gen, sweep
runners) is:

    parse_quantity(q)   -> float   SI value from a {value, unit} mapping
    load_aircraft(path) -> AircraftParams   recursive attribute-access tree
    reynolds(...)       -> float   Re = V c / nu
    dynamic_pressure(...) -> float q_inf = 0.5 rho V^2

Run ``python -m geometry.units`` (or ``python geometry/units.py``) for an
ASCII summary table of aircraft.yaml in SI — a quick sanity check that the
master file parses and the conversions look physically plausible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

# =============================================================================
#  Unit conversion table
# =============================================================================
#  Every entry is a pure multiplicative factor tag -> SI. This works because
#  all accepted tags are offset-free: kelvin is already the SI temperature
#  unit (no Celsius/Fahrenheit tags are accepted, deliberately — an offset
#  conversion hiding inside a multiply-only table is how silent bugs happen).
#  Passthrough tags (kgpm3, m2ps, mps, m, ...) are listed explicitly rather
#  than special-cased so that an unrecognized tag is ALWAYS a hard error.
UNIT_TO_SI: Dict[str, float] = {
    # --- length -> m ---
    "m":     1.0,
    "mm":    1.0e-3,
    "in":    0.0254,                 # international inch (exact)
    "ft":    0.3048,                 # international foot (exact)
    # --- area -> m^2 ---
    "m2":    1.0,
    "ft2":   0.3048 ** 2,            # 0.09290304 exactly
    # --- speed -> m/s ---
    "mps":   1.0,
    "mph":   0.44704,                # statute mph (exact)
    "kt":    1852.0 / 3600.0,        # international knot (exact)
    # --- angle -> rad ---
    "deg":   math.pi / 180.0,
    "rad":   1.0,                    # accepted for symmetry; rarely used in YAML
    # --- temperature -> K (absolute only; no offset units accepted) ---
    "K":     1.0,
    # --- mass -> kg ---
    "kg":    1.0,
    "lb":    0.45359237,             # avoirdupois pound (exact)
    "slug":  14.59390294,            # lbf s^2/ft; listed in aircraft.yaml header
    # --- density -> kg/m^3 (passthrough; ISA values are entered in SI) ---
    "kgpm3": 1.0,
    # --- kinematic viscosity -> m^2/s (passthrough) ---
    "m2ps":  1.0,
}


# =============================================================================
#  Quantity parsing
# =============================================================================
def parse_quantity(q: Dict[str, Any], key_path: str = "") -> float:
    """Convert one ``{value: <num>, unit: "<tag>"}`` mapping to a float in SI.

    ``key_path`` is optional context (e.g. ``"wing.span"``) threaded through
    by the recursive loader so that a bad tag deep inside aircraft.yaml is
    reported with its full location instead of an anonymous dict repr.

    Raises ValueError on a malformed mapping, a non-numeric value, or an
    unknown unit tag — never silently passes a suspect quantity through.
    """
    # Where-am-I prefix for every error message; empty when called standalone.
    where = f" at '{key_path}'" if key_path else ""

    # Shape check: must be a mapping carrying both required keys. Anything
    # else reaching this function indicates a YAML authoring mistake.
    if not isinstance(q, dict) or "value" not in q or "unit" not in q:
        raise ValueError(
            f"Quantity{where} must be a mapping with 'value' and 'unit' keys, "
            f"got: {q!r}"
        )

    # The value must be a real number; bool is excluded explicitly because
    # bool is a subclass of int in Python and True would convert to 1.0.
    value = q["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Quantity{where} has non-numeric value: {value!r}")

    # Tag lookup. Unknown tags name the offending key path AND list the
    # accepted tags so the YAML author can fix the file without reading code.
    unit = q["unit"]
    if not isinstance(unit, str) or unit not in UNIT_TO_SI:
        raise ValueError(
            f"Unknown unit tag {unit!r}{where}. "
            f"Accepted tags: {', '.join(sorted(UNIT_TO_SI))}"
        )

    return float(value) * UNIT_TO_SI[unit]


# =============================================================================
#  Recursive YAML -> attribute tree
# =============================================================================
def _is_quantity(node: Any) -> bool:
    """True when a YAML node is a unit-tagged quantity mapping.

    Detection is by the presence of BOTH 'value' and 'unit' keys: ordinary
    parameter groups in aircraft.yaml never use those names, so this test is
    unambiguous for this schema (and parse_quantity re-validates anyway).
    """
    return isinstance(node, dict) and "value" in node and "unit" in node


def _convert_node(node: Any, key_path: str) -> Any:
    """Convert one YAML node to its SI runtime form, recursing as needed.

    Rules (the cross-module contract):
      * {value, unit} mapping  -> float in SI
      * other mapping          -> AircraftParams (attribute access)
      * list                   -> list, each element converted (mappings
                                  recurse, so conditions.cruise_points becomes
                                  a list of AircraftParams)
      * None                   -> None  (unknowns stay visibly unknown)
      * scalar (num/str/bool)  -> passthrough unchanged
    """
    # None first: YAML 'null' marks a deliberately-unresolved parameter
    # (e.g. vertical_tail.area) and must survive untouched so downstream
    # code can detect and refuse to use it.
    if node is None:
        return None

    # Unit-tagged quantity -> SI float. This consumes the mapping entirely;
    # the original {value, unit} form remains reachable via .raw.
    if _is_quantity(node):
        return parse_quantity(node, key_path)

    # Plain mapping -> nested parameter object mirroring the YAML structure.
    if isinstance(node, dict):
        return AircraftParams(node, key_path)

    # Lists recurse element-wise with an indexed key path so errors inside
    # e.g. cruise_points[2].speed are reported at their exact location.
    if isinstance(node, list):
        return [
            _convert_node(item, f"{key_path}[{i}]") for i, item in enumerate(node)
        ]

    # Dimensionless scalars (taper ratio, CL values, file-name strings, ...)
    # pass through verbatim.
    return node


class AircraftParams:
    """Read-mostly attribute view of one mapping level of aircraft.yaml.

    Built recursively by load_aircraft(): every nested mapping becomes
    another AircraftParams, every {value, unit} leaf becomes an SI float,
    so client code reads ``ac.wing.chord_root`` and gets meters with zero
    ceremony. The original un-converted mapping for THIS level is kept on
    ``.raw`` for provenance, diffing, and re-serialization.
    """

    # Attribute names claimed by the class itself; a YAML key colliding with
    # one of these would silently shadow infrastructure, so it is rejected.
    _RESERVED = ("raw",)

    def __init__(self, data: Dict[str, Any], key_path: str = "") -> None:
        # Guard against schema/implementation collisions up front — better a
        # loud failure at load time than a shadowed .raw discovered in a
        # post-processing script three phases later.
        for reserved in self._RESERVED:
            if reserved in data:
                raise ValueError(
                    f"YAML key {reserved!r} at '{key_path or '<root>'}' "
                    f"collides with a reserved AircraftParams attribute"
                )

        # Convert and bind each child. setattr (rather than a backing dict
        # plus __getattr__) keeps attribute access plain-Python: tab
        # completion, dataclass-like repr, and AttributeError with the real
        # missing name all work without extra machinery.
        for key, value in data.items():
            child_path = f"{key_path}.{key}" if key_path else key
            setattr(self, key, _convert_node(value, child_path))

        # Original mapping preserved verbatim (same object, not a copy — the
        # YAML loader built it fresh, so nothing else mutates it).
        self.raw: Dict[str, Any] = data

    def __repr__(self) -> str:
        # Compact field listing; .raw excluded to keep reprs readable when
        # debugging nested structures in a REPL.
        keys = [k for k in vars(self) if k != "raw"]
        return f"AircraftParams({', '.join(keys)})"


def load_aircraft(path: Union[str, Path]) -> AircraftParams:
    """Load aircraft.yaml and return the SI-converted parameter tree.

    The returned object mirrors the YAML nesting via attributes:
    ``ac.wing.span`` is meters, ``ac.vg_defaults.vane_incidence`` is radians,
    ``ac.wing.taper_ratio`` passes through dimensionless, ``ac.raw`` holds
    the original parsed dict. Conversion errors carry the full key path.
    """
    path = Path(path)
    # safe_load only: the master parameter file must never be able to
    # instantiate arbitrary Python objects.
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # An empty or non-mapping top level means the file is broken, not merely
    # sparse — fail loudly with the filename.
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a YAML mapping, got {type(data).__name__}")

    return AircraftParams(data)


# =============================================================================
#  Flow helpers used by later phases (Re grids, first-cell-height sizing,
#  force-coefficient normalization)
# =============================================================================
def reynolds(speed_mps: float, chord_m: float, nu: float) -> float:
    """Chord Reynolds number Re = V c / nu.

    ``nu`` is kinematic viscosity in m^2/s (atmosphere.kinematic_viscosity
    from aircraft.yaml at ISA sea level). All inputs SI; no unit handling
    here by design — conversion happens once, at load.
    """
    return speed_mps * chord_m / nu


def dynamic_pressure(rho: float, speed_mps: float) -> float:
    """Freestream dynamic pressure q = 0.5 rho V^2 in Pa (SI inputs)."""
    return 0.5 * rho * speed_mps ** 2


# =============================================================================
#  CLI sanity check: dump aircraft.yaml as a flat SI table
# =============================================================================
def _walk_raw(node: Any, key_path: str, rows: List[tuple]) -> None:
    """Flatten the raw YAML tree into (path, si_value, original) rows.

    Walks the RAW dict (not the converted tree) so each row can show both
    the SI result and the source value+unit it came from — the fastest way
    to eyeball a transcription error against the reference documents.
    """
    if _is_quantity(node):
        # Leaf quantity: convert and record alongside its source form.
        si = parse_quantity(node, key_path)
        rows.append((key_path, f"{si:.6g}", f"{node['value']} {node['unit']}"))
    elif isinstance(node, dict):
        # Mapping: descend into each child with an extended path.
        for key, value in node.items():
            child = f"{key_path}.{key}" if key_path else key
            _walk_raw(value, child, rows)
    elif isinstance(node, list):
        # Lists get indexed paths so cruise points line up readably.
        for i, item in enumerate(node):
            _walk_raw(item, f"{key_path}[{i}]", rows)
    elif node is None:
        rows.append((key_path, "None", "(unresolved)"))
    else:
        # Dimensionless scalar or string: shown as-is, no source column.
        rows.append((key_path, str(node), "-"))


def _main() -> int:
    """Print an ASCII SI summary of aircraft.yaml (quick sanity CLI)."""
    # Resolve the master file relative to the repo root regardless of CWD —
    # this module lives at <root>/geometry/units.py.
    yaml_path = Path(__file__).resolve().parents[1] / "aircraft.yaml"
    ac = load_aircraft(yaml_path)

    # Flatten everything for the table body.
    rows: List[tuple] = []
    _walk_raw(ac.raw, "", rows)

    # Fixed-width, plain-ASCII layout — safe on the default Windows console.
    print("=" * 92)
    print("aircraft.yaml SI summary  ({})".format(yaml_path))
    print("=" * 92)
    print(f"{'parameter':<52} {'SI value':>16}   source")
    print("-" * 92)
    for path, si, src in rows:
        print(f"{path:<52} {si:>16}   {src}")
    print("-" * 92)

    # A couple of derived numbers that exercise the helpers and give an
    # immediate plausibility check (approach-condition Re on the MAC should
    # land in the mid-3-millions for this airframe).
    v_app = ac.conditions.approach_speed
    re_app = reynolds(v_app, ac.wing.mac, ac.atmosphere.kinematic_viscosity)
    q_app = dynamic_pressure(ac.atmosphere.density, v_app)
    print(f"{'derived: Re(approach, MAC)':<52} {re_app:>16.4g}   V*c/nu")
    print(f"{'derived: q(approach) [Pa]':<52} {q_app:>16.6g}   0.5*rho*V^2")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
