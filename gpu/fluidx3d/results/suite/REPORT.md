# VG comparison suite -- analysis from per-case force CSVs

Slice mode, 14 deg AoA, settled-window averages (first 40% of samples discarded).
Trust DELTAS against the clean baseline; absolute values carry
shared coarse-lattice bias. Stall-speed projections are an
indicative single-AoA proxy, not a measured CLmax ratio.

## 80mph (clean baseline: Cl=1.9462 Cd=0.2956 buffet=0.6747, 34 samples)

| design | mean Cl | dCl | dCl % | proj. stall speed | mean Cd | dCd | buffet |
|---|---|---|---|---|---|---|---|
| clean | 1.9462 | -- | -- | 80 mph (ref) | 0.2956 | -- | 0.6747 |
| vg08mm | 1.6221 | -0.3240 | -16.7% | ~87.6 mph | 0.4017 | +0.1061 | 0.6558 |
| vg12mm | (incomplete) | | | | | | |
| vg16mm | (incomplete) | | | | | | |

## 60mph: no complete clean baseline yet
