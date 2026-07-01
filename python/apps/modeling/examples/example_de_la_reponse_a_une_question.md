# Example: Creating a multi-shot acquisition modeling script

## Task

From `acoustic_one_shot_unstruct_sep.py`, create `acquisition_acoustic_modeling.py` that:

- Takes a **decision file** as input (geometry for all shots)
- Loops over shots, running a forward acoustic simulation for each
- Stores each shot's seismogram as a **SEG-Y file** with geometry headers
- Supports running a **subset** of shots via `--first-shot` / `--last-shot`

---

## Approach

### Files read for context

| File | Purpose |
|---|---|
| `acoustic_one_shot_unstruct_sep.py` | Reference implementation (single shot) |
| `utilities/acquisition/decision_reader.py` | `read_decision_file()` API |
| `utilities/acquisition/create_equispaced_segy.py` | SEG-Y write helpers |
| `utilities/acquisition/SegyAcquisition.py` | SEG-Y reader pattern |

### Key design decisions

1. **Mesh and solver are built once** before the shot loop — no reinitialization per shot.
2. `solver.resetWaveField()` is called at the start of each shot.
3. Shot range `[first, last]` is 1-based and inclusive; defaults to all shots.
4. Geometry is read from the decision file via `read_decision_file()` with
   `scalco=1, scalel=1` (coordinates stored in metres as-is).
5. The SEG-Y writer is kept **self-contained** in the new script (no dependency on
   `create_equispaced_segy.py`) so that it writes **real float32 trace data**
   instead of placeholder zeros.
6. Output files are separate from the geometry placeholder SEG-Y files:
   `{out}_shot_{N:05d}.segy` in `--output-dir`.

### `read_decision_file` output keys

```
d_src : { "src1": [x,y,z],  "src2": [x,y,z],  ... }
d_rcv : { "rcv_1": [[x,y,z], ...],  "rcv_2": [[x,y,z], ...],  ... }
n     : total number of shots
```

---

## Result

**Script**: [`acquisition_acoustic_modeling.py`](acquisition_acoustic_modeling.py)

### Usage

```bash
python acquisition_acoustic_modeling.py \
    --vp-file  sep_models/vp_layered.H \
    --decision acquisition/survey \
    [--rho-file sep_models/rho.H] \
    [--first-shot 1] [--last-shot 10] \
    [--f0 10] [--tmax 2.0] [--order 2] \
    [--output-dir seismoTraces] [--out survey] \
    [--snapshots] [--snapshot-interval 100]
```

### Shot loop (pseudocode)

```
build mesh once
build solver once
read d_src, d_rcv, n_shots from decision file

for shot_idx in [first_shot .. last_shot]:
    src  = d_src["src{shot_idx}"]
    rcvs = d_rcv["rcv_{shot_idx}"]

    solver.resetWaveField()
    solver.updateSourceAndReceiversFromCoords([src], rcvs)
    solver.evaluateSource()

    for t in range(n_steps):
        solver.execute(t)

    seismogram = solver.getPressureAtReceivers()   # (n_samples, n_rcv)
    write_segy_with_data(output_dir / "survey_shot_NNNNN.segy", ...)
```

### Output per shot

```
seismoTraces/
  survey_shot_00001.segy   # SEG-Y with geometry headers + float32 traces
  survey_shot_00002.segy
  ...
  snapshots/               # (optional, --snapshots)
    shot00001_step000000.png
    shot00001_step000100.png
    ...
```
