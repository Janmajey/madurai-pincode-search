# Mutation test

Deleted 592 items from MUM-METRO (10%), added 1817 'Festive Gifting' items (5%) with a new category, new titles and their own price/zone mix.
The new items landed in 51 of 256 partitions; the three biggest took 361, 111, 103.

| Stage | ~100% | ~10% | ~1% | ~0.1% | handful | new category | Largest partition | Size std | Mean item-centroid similarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before | 0.972 / 1.04 ms | 0.960 / 0.53 ms | 1.000 / 0.21 ms | 1.000 / 0.09 ms | 1.000 / 0.08 ms | n/a | 401 | 74.7 | 0.844 |
| after (incremental) | 0.968 / 1.17 ms | 0.970 / 0.54 ms | 1.000 / 0.20 ms | 1.000 / 0.08 ms | 1.000 / 0.07 ms | 1.000 / 0.27 ms | 522 | 81.4 | 0.838 |
| after (rebuilt) | 0.963 / 0.99 ms | 0.957 / 0.47 ms | 1.000 / 0.22 ms | 1.000 / 0.09 ms | 1.000 / 0.07 ms | 1.000 / 0.23 ms | 420 | 75.7 | 0.843 |

Cells are planner recall@10 / p50 latency.
