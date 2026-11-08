# Acquisition test data

Used by `tests/unitTests/test_Acquisition.py`.

| File | Content |
|---|---|
| `shot_1.sgy`, `shot_2.sgy` | 2 shots of 300 receivers; integer coordinates with scalco = scalel = -10 (divide by 10): sources at x = 399.1 and 599.1 m, y = 499.1 m, z = 20.1 m (`sdepth`), receivers at z = 30.1 m (`gelev`) |
| `decision.*` | Decision files of the two shots: raw integers, no scalar words; source words 22, 23, 14, receiver words 24, 25, 13 |

The decision files were created from the SEG-Y files with (from `FUnTiDES-App/python`):

```bash
python3 apps/acquisition/decision_create.py \
    --from-segy tests/data/acquisition/shot_1.sgy tests/data/acquisition/shot_2.sgy \
    --out tests/data/acquisition/decision --index-words 22,23,14
```

The SEG-Y words read are 19, 20, 13 for the source (sx, sy, sdepth) and 21, 22, 11 for
the receivers (gx, gy, gelev); `decision.mapped.list` gives the SEG-Y paths relative to
`FUnTiDES-App/python`.
