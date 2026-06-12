#!/bin/bash
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
