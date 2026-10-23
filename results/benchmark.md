# Benchmark results

Catalog: 36932 items, 300 dims, 256 partitions.

## Planner settings (measured)

- base nprobe: 16 (smallest nprobe with unfiltered recall ≥ 0.97)
- partition-restricted coverage: 20% of the expected matches (smallest value with recall ≥ 0.97 in every band from 2% to 50%)
- decision rule (estimated selectivity → strategy), fastest strategy with recall ≥ 0.97 in each band:
  - 0.00% to 5.00%: exact scan
  - 5.00% to 50.00%: partition-restricted
  - 50.00% to 100.00%: adaptive post-filter

## nprobe sweep (no filter)

| nprobe | Recall@10 | p50 ms |
| --- | --- | --- |
| 1 | 0.636 | 0.07 |
| 2 | 0.793 | 0.13 |
| 4 | 0.883 | 0.27 |
| 8 | 0.941 | 0.54 |
| 16 | 0.983 | 0.95 |
| 32 | 0.994 | 1.93 |
| 64 | 0.997 | 3.74 |
| 128 | 1.000 | 6.84 |

## Coverage sweep for partition-restricted search (2% to 50% selectivity)

| Coverage | Worst band recall | Mean p50 ms |
| --- | --- | --- |
| 5% | 0.933 | 0.29 |
| 10% | 0.945 | 0.38 |
| 15% | 0.964 | 0.45 |
| 20% | 0.972 | 0.64 |
| 25% | 0.976 | 0.77 |
| 30% | 0.986 | 0.85 |
| 40% | 0.991 | 1.06 |

## Crossover (calibration set, bucketed by true selectivity)

| Selectivity | Queries | exact scan p50 ms (recall) | adaptive post-filter p50 ms (recall) | partition-restricted p50 ms (recall) | Pick (fastest with recall 0.97 or more, a tie within 15% goes to better recall) |
| --- | --- | --- | --- | --- | --- |
| 0.00% to 0.05% | 343 | 0.06 (1.00) | 0.38 (1.00) | 0.18 (1.00) | exact scan |
| 0.05% to 0.10% | 90 | 0.05 (1.00) | 0.32 (1.00) | 0.17 (1.00) | exact scan |
| 0.10% to 0.20% | 104 | 0.05 (1.00) | 0.26 (1.00) | 0.16 (0.97) | exact scan |
| 0.20% to 0.50% | 197 | 0.07 (1.00) | 0.30 (1.00) | 0.14 (0.93) | exact scan |
| 0.50% to 1.00% | 88 | 0.10 (1.00) | 0.37 (1.00) | 0.16 (0.90) | exact scan |
| 1.00% to 2.00% | 126 | 0.13 (1.00) | 0.36 (1.00) | 0.16 (0.95) | exact scan |
| 2.00% to 3.50% | 111 | 0.20 (1.00) | 0.41 (1.00) | 0.17 (0.98) | partition-restricted |
| 3.50% to 5.00% | 42 | 0.31 (1.00) | 0.50 (1.00) | 0.21 (0.96) | exact scan |
| 5.00% to 7.50% | 90 | 0.60 (1.00) | 0.80 (1.00) | 0.25 (0.97) | partition-restricted |
| 10.00% to 15.00% | 52 | 1.21 (1.00) | 0.96 (0.99) | 0.34 (0.97) | partition-restricted |
| 20.00% to 30.00% | 29 | 2.42 (1.00) | 1.18 (0.99) | 0.59 (0.98) | partition-restricted |
| 30.00% to 50.00% | 31 | 2.31 (1.00) | 0.92 (0.98) | 1.03 (0.99) | partition-restricted |
| 50.00% to 75.00% | 28 | 2.40 (1.00) | 1.02 (0.99) | 1.67 (1.00) | adaptive post-filter |
| 75.00% to 100.00% | 144 | 2.52 (1.00) | 1.05 (0.97) | 2.49 (0.99) | adaptive post-filter |

## Results by selectivity bucket

| Bucket | Method | Recall@10 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- |
| ~100% | post-filter (baseline) | 0.954 | 1.24 | 1.68 |
| ~100% | pre-filter (baseline) | 1.000 | 2.66 | 3.15 |
| ~100% | exact scan | 1.000 | 2.91 | 3.46 |
| ~100% | adaptive post-filter | 0.976 | 1.44 | 2.17 |
| ~100% | partition-restricted | 0.982 | 1.95 | 2.52 |
| ~100% | planner | 0.976 | 1.19 | 1.95 |
| ~10% | post-filter (baseline) | 0.738 | 1.02 | 2.00 |
| ~10% | pre-filter (baseline) | 1.000 | 1.13 | 2.26 |
| ~10% | exact scan | 1.000 | 1.18 | 2.17 |
| ~10% | adaptive post-filter | 1.000 | 1.17 | 1.74 |
| ~10% | partition-restricted | 0.966 | 0.45 | 0.65 |
| ~10% | planner | 0.966 | 0.50 | 0.73 |
| ~1% | post-filter (baseline) | 0.336 | 1.04 | 1.45 |
| ~1% | pre-filter (baseline) | 1.000 | 0.16 | 0.27 |
| ~1% | exact scan | 1.000 | 0.17 | 0.28 |
| ~1% | adaptive post-filter | 1.000 | 0.44 | 0.58 |
| ~1% | partition-restricted | 0.938 | 0.20 | 0.30 |
| ~1% | planner | 1.000 | 0.21 | 0.33 |
| ~0.1% | post-filter (baseline) | 0.112 | 0.95 | 1.32 |
| ~0.1% | pre-filter (baseline) | 1.000 | 0.06 | 0.09 |
| ~0.1% | exact scan | 1.000 | 0.06 | 0.12 |
| ~0.1% | adaptive post-filter | 1.000 | 0.35 | 0.46 |
| ~0.1% | partition-restricted | 0.976 | 0.19 | 0.38 |
| ~0.1% | planner | 1.000 | 0.09 | 0.17 |
| handful | post-filter (baseline) | 0.043 | 0.93 | 1.31 |
| handful | pre-filter (baseline) | 1.000 | 0.05 | 0.11 |
| handful | exact scan | 1.000 | 0.05 | 0.11 |
| handful | adaptive post-filter | 1.000 | 0.36 | 0.44 |
| handful | partition-restricted | 1.000 | 0.18 | 0.36 |
| handful | planner | 1.000 | 0.08 | 0.12 |

## What the planner picked

| Bucket | exact scan | adaptive post-filter | partition-restricted |
| --- | --- | --- | --- |
| ~100% | 0 | 49 | 1 |
| ~10% | 1 | 0 | 49 |
| ~1% | 50 | 0 | 0 |
| ~0.1% | 50 | 0 | 0 |
| handful | 50 | 0 | 0 |

## Per-zone recall (zone-only filters, 20 queries per zone)

| Method | Traffic-weighted recall | Plain average | Worst zone (items) | Worst recall |
| --- | --- | --- | --- | --- |
| post-filter (baseline) | 0.594 | 0.271 | LEH (10) | 0.000 |
| planner | 0.985 | 0.995 | BLR-METRO (4310) | 0.970 |
