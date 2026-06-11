# tools/xfoil — XFOIL 6.99 binary provenance

The XFOIL executable used by `validation/xfoil_polar.py` is NOT committed
(binaries stay out of the repository; `tools/` is gitignored except for this
file). Re-download it with the steps below before running the Phase-1
validation sweep on a fresh clone.

## Exact source

| field        | value                                                              |
|--------------|--------------------------------------------------------------------|
| Package      | XFOIL 6.99, official Windows executable build                       |
| Author       | Mark Drela / Harold Youngren, MIT                                   |
| URL          | https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip            |
| Landing page | https://web.mit.edu/drela/Public/web/xfoil/                         |
| SHA-256      | `e13e8fe5cc38d8ac2626e9d3b17643bdcfaa63791619f042afdaa7cd103bcb08`  |
| License      | GNU GPL (stated in the program banner and the bundled source zip)   |
| Downloaded   | 2026-06-10                                                          |

Archive contents: `xfoil.exe` (the only file the pipeline needs), `pplot.exe`,
`pxplot.exe` (polar/airfoil plotters, unused here), `Xfoil699src.zip`
(Fortran source, kept for license compliance and future WSL builds).

## Re-download steps (Windows, from the repo root)

```powershell
curl.exe -L -o tools\xfoil\XFOIL6.99.zip https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip
# Verify integrity against the hash above before trusting the binary:
Get-FileHash tools\xfoil\XFOIL6.99.zip -Algorithm SHA256
Expand-Archive tools\xfoil\XFOIL6.99.zip -DestinationPath tools\xfoil -Force
```

Sanity check (must print the 6.99 banner and exit 0):

```powershell
echo QUIT | tools\xfoil\xfoil.exe
```

Only the official MIT distribution above is acceptable — do not substitute
third-party rebuilds. If the direct zip URL ever moves, locate the new link
on the landing page (same site) and update this file with the new hash.

## Alternate binary locations

`validation/xfoil_polar.py` resolves the executable in this order:

1. `--xfoil <path>` command-line flag
2. `XFOIL_BIN` environment variable
3. `tools/xfoil/xfoil.exe` (this directory)
4. `wsl xfoil` fallback (an XFOIL installed inside WSL2 Ubuntu, e.g. built
   from the bundled `Xfoil699src.zip` or `apt install xfoil`)
