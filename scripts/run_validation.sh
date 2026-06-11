#!/usr/bin/env bash
# M1 acceptance entry point (spec section 5): one command runs the Phase-1
# clean-section validation end to end on the WSL2/Ubuntu side. This wrapper
# only prepares the solver environment and delegates everything (XFOIL
# baseline, case build, solve, gate report) to scripts/run_validation.py.
#
# Usage:  ./scripts/run_validation.sh [--re 3e6] [--level 0] [--jobs 2] ...
#         (all flags pass through to run_validation.py unchanged)
set -u

# Resolve the repository root from this script's own location so the command
# works from any working directory (mirrors the Python-side bootstrap).
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Source the pinned ESI OpenFOAM v2506 environment when present. Probing a
# small list keeps the wrapper usable on a host where a newer pinned release
# replaces v2506 later; first hit wins. Missing bashrc is NOT fatal - the
# Python driver probes for simpleFoam itself and dry-runs gracefully.
for bashrc in \
    /usr/lib/openfoam/openfoam2506/etc/bashrc \
    "${HOME}/OpenFOAM/OpenFOAM-v2506/etc/bashrc"
do
    if [ -f "${bashrc}" ]; then
        # OpenFOAM's bashrc references unset variables; relax -u around it.
        set +u
        # shellcheck disable=SC1090
        . "${bashrc}"
        set -u
        break
    fi
done

# Hand off to the Python driver; exec keeps the exit status transparent to
# CI and the milestone checklist. Ubuntu ships python3; the python fallback
# covers a git-bash invocation on the Windows host.
PY="$(command -v python3 || command -v python)"
exec "${PY}" "${REPO}/scripts/run_validation.py" "$@"
