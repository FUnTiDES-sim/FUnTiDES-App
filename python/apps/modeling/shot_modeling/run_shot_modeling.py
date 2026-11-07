"""
Acoustic shot modelling on a SEP velocity model, 2-D or 3-D, one or several shots.

Workflow
--------
1. Model: read the SEP Vp (and optional Rho) file. Its dimension sets the case:
   n3 = 1 -> 2-D (axis 1 = x, axis 2 = z), otherwise 3-D (axes x, y, z); z is the
   depth (seismic convention: 0 at the surface, positive downwards).
2. Mesh and time step (``makutu.mesh.suggest_mesh``): element size from the shortest
   wavelength (``--ppw`` GLL points per wavelength) or ``--elem-size``, dt from ``--cfl``.
   Free surface on top, absorbing edges elsewhere (optional sponge of ``--sponge`` m).
3. Acquisition, either
   - a decision file (``--decision``), as written by
     ``python -m makutu.acquisition.create_equispaced_segy``, or
   - a line of ``--n-shots`` sources along x (one at ``--src-x`` by default) and
     receivers every ``--rcv-dx``, at y = ly / 2 in 3-D.
4. Shot loop on the same mesh and solver: one SEG-Y (or HDF5) file per shot, and
   with ``--plot`` a figure per shot (model section, pressure snapshot, gather).

Usage
-----
    python run_shot_modeling.py --vp-file Vp.H [--rho-file Rho.H] [--f0 10 --order 2]
        [--decision acquisition/survey | --n-shots 5 --src-depth 10 --rcv-depth 20 --rcv-dx 20]
        [--tmax 2] [--format segy|hdf5] [--plot] [--output-dir runs/shots]
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np

# kokkos MUST be initialized before any pyfuntides constructor is called
import kokkos
kokkos.initialize()

import pyfuntides.solver as FTSolver

from makutu.mesh import mesh_from_sep
from makutu.solvers import AcousticSolver
from makutu.acquisition import Acquisition, DecisionAcquisition
from makutu.acquisition.create_equispaced_segy import write_segy_shot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--vp-file", required=True, help="SEP header of the Vp model [m/s]")
    g.add_argument("--rho-file", default=None, help="SEP header of the density (default 1)")

    g = p.add_argument_group("mesh and time")
    g.add_argument("--f0", type=float, default=10.0, help="Ricker peak frequency [Hz] (default 10)")
    g.add_argument("--order", type=int, default=2,
                   help="polynomial order (default 2; up to 3 in 3-D, up to the compiled order in 2-D)")
    g.add_argument("--ppw", type=float, default=6.0, help="GLL points per shortest wavelength (default 6)")
    g.add_argument("--elem-size", type=float, default=None, help="element size [m] (overrides --ppw)")
    g.add_argument("--cfl", type=float, default=0.3, help="CFL number (default 0.3)")
    g.add_argument("--tmax", type=float, default=None, help="recorded time [s] (default 2 lz / vp_min)")
    g.add_argument("--dt-seismo", type=float, default=None, help="seismogram sampling [s] (default dt)")
    g.add_argument("--sponge", type=float, default=0.0,
                   help="sponge width on the absorbing edges [m] (default 0: absorbing condition only; "
                        "the sponge does better from about two wavelengths)")

    g = p.add_argument_group("acquisition (decision file)")
    g.add_argument("--decision", default=None, help="decision file rootname (e.g. acquisition/survey)")
    for name, word, what in (("sx", 22, "source x"), ("sy", 23, "source y"), ("sz", 14, "source z"),
                             ("rx", 24, "receiver x"), ("ry", 25, "receiver y"), ("rz", 13, "receiver z")):
        g.add_argument(f"--word-{name}", type=int, default=word, help=f"{what} word (default {word})")
    g.add_argument("--first-shot", type=int, default=1, help="first shot, 1-based (default 1)")
    g.add_argument("--last-shot", type=int, default=None, help="last shot (default: all)")

    g = p.add_argument_group("acquisition (line along x, without --decision)")
    g.add_argument("--n-shots", type=int, default=1, help="number of sources (default 1)")
    g.add_argument("--src-x", type=float, default=None, help="source x for one shot [m] (default lx / 2)")
    g.add_argument("--src-xmin", type=float, default=None, help="first source x [m] (default 10%% of lx)")
    g.add_argument("--src-xmax", type=float, default=None, help="last source x [m] (default 90%% of lx)")
    g.add_argument("--src-depth", type=float, default=10.0, help="source z [m] (default 10)")
    g.add_argument("--rcv-depth", type=float, default=20.0, help="receiver z [m] (default 20)")
    g.add_argument("--rcv-dx", type=float, default=None,
                   help="receiver spacing [m], from rcv_dx / 2 to lx (default: element size)")

    g = p.add_argument_group("outputs")
    g.add_argument("--output-dir", default="shots", help="output directory (default shots)")
    g.add_argument("--out", default=None, help="file rootname (default: decision rootname or 'shot')")
    g.add_argument("--format", choices=("segy", "hdf5"), default="segy", help="gather format (default segy)")
    g.add_argument("--plot", action="store_true", help="figure per shot: model, snapshot, gather")
    g.add_argument("--snap-time", type=float, default=None, help="snapshot time [s] (default tmax / 3)")
    return p.parse_args()


def build_acquisition(args, lengths, dim, h):
    """Acquisition from the decision file or a line of sources and receivers along x."""
    if args.decision is not None:
        return DecisionAcquisition(
            decision_filename=args.decision,
            swords=[args.word_sx, args.word_sy, args.word_sz],
            rwords=[args.word_rx, args.word_ry, args.word_rz],
            scalco=1, scalel=1)
    lx = lengths[0]
    y = lengths[1] / 2.0 if dim == 3 else 0.0
    if args.n_shots == 1:
        xs = [args.src_x if args.src_x is not None else lx / 2.0]
    else:
        x0 = args.src_xmin if args.src_xmin is not None else 0.1 * lx
        x1 = args.src_xmax if args.src_xmax is not None else 0.9 * lx
        xs = np.linspace(x0, x1, args.n_shots)
    dx = args.rcv_dx if args.rcv_dx is not None else h
    sources = [[float(x), y, args.src_depth] for x in xs]
    receivers = [[float(x), y, args.rcv_depth] for x in np.arange(dx / 2.0, lx, dx)]
    return Acquisition(sources=sources, receivers=receivers)


def model_section(solver, mesh, dim):
    """Vp section (z down): the model in 2-D, the x-z plane at y = ly / 2 in 3-D."""
    vp = solver.getVelocityModel()
    if dim == 2:
        return vp.reshape(mesh.ez, mesh.ex)
    return vp.reshape(mesh.ez, mesh.ey, mesh.ex)[:, mesh.ey // 2, :]


def pressure_section(solver, dim):
    p = solver.getPressureField()                 # (nz, nx) or (nz, ny, nx), z down
    return p if dim == 2 else p[:, p.shape[1] // 2, :]


def plot_shot(path, model, snapshot, snap_time, gather, dt_rec, src, rcv, lx, lz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 11))
    km = [0, lx / 1e3, lz / 1e3, 0]
    im = ax[0].imshow(model, extent=km, aspect="auto", cmap="viridis")
    ax[0].plot(src[0] / 1e3, src[2] / 1e3, "r*", ms=12)
    ax[0].set(title="Vp (m/s)", xlabel="x (km)", ylabel="z (km)")
    fig.colorbar(im, ax=ax[0])
    if snapshot is not None:
        clip = np.percentile(np.abs(snapshot), 99.5) or 1.0
        ax[1].imshow(snapshot, extent=km, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip)
        ax[1].set(title=f"Pressure at t = {snap_time:g} s", xlabel="x (km)", ylabel="z (km)")
    clip = np.percentile(np.abs(gather), 99) or 1.0
    x = np.asarray(rcv)[:, 0] / 1e3
    ax[2].imshow(gather, extent=[x.min(), x.max(), gather.shape[0] * dt_rec, 0], aspect="auto",
                 cmap="gray", vmin=-clip, vmax=clip)
    ax[2].set(title="Shot gather", xlabel="receiver x (km)", ylabel="time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    m = mesh_from_sep(args.vp_file, args.f0, args.order, rho_file=args.rho_file, ppw=args.ppw, cfl=args.cfl,
                      elem_size=args.elem_size)
    mesh, dim, lengths, dt, vp_min = m.mesh, m.dim, m.lengths, m.dt, m.vp_min
    lx, lz = lengths[0], lengths[-1]
    h = lx / mesh.ex

    tmax = args.tmax if args.tmax is not None else 2.0 * lz / vp_min
    sponge = args.sponge
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU,
                            mesh_type=FTSolver.MeshType.UNSTRUCT,       # SEP model: ModelUnstruct
                            model_location=FTSolver.ModelLocationType.ONELEMENTS,
                            sponge_size=[sponge, sponge, sponge], surface_sponge=True,
                            dt=dt, minTime=0.0, maxTime=tmax, dtSeismo=args.dt_seismo or dt,
                            sourceType="ricker2", sourceFreq=args.f0, **mesh.solver_kwargs())
    solver.initialize()
    n_steps = solver.n_time_steps
    dt_rec = max(1, int(round(solver.dtSeismo / dt))) * dt   # sampling actually recorded by the solver
    print(f"Time  : dt {dt * 1e3:.3f} ms, {n_steps} steps for {tmax:g} s; seismograms every "
          f"{dt_rec * 1e3:.3f} ms; sponge {sponge:g} m")

    acquisition = build_acquisition(args, lengths, dim, h)
    n_shots = len(acquisition.shots)
    first, last = args.first_shot, args.last_shot or n_shots
    if not 1 <= first <= last <= n_shots:
        raise SystemExit(f"invalid shot range [{first}, {last}] for {n_shots} shots")
    root = args.out or (os.path.basename(args.decision) if args.decision else "shot")
    os.makedirs(args.output_dir, exist_ok=True)
    snap_time = args.snap_time if args.snap_time is not None else tmax / 3.0
    snap_step = int(round(snap_time / dt))
    print(f"Shots : {first} to {last} of {n_shots}")

    for i in range(first, last + 1):
        shot = acquisition.shots[i - 1]
        src, rcv = shot.getSourceCoords()[0], shot.getReceiverCoords()
        print(f"\nShot {i}: source ({src[0]:.1f}, {src[1]:.1f}, {src[2]:.1f}), {len(rcv)} receivers")
        solver.resetWaveField()
        solver.updateSourceAndReceiversFromCoords([src], rcv)
        solver.evaluateSource()
        snapshot = None
        for t in range(n_steps):
            solver.execute(t)
            if args.plot and t == snap_step:
                snapshot = pressure_section(solver, dim)
            if t % max(1, n_steps // 10) == 0:
                print(f"  step {t:6d} / {n_steps}  (t = {t * dt:.3f} s)", end="\r", flush=True)
        print()
        gather = solver.getPressureAtReceivers()           # (n_samples, n_receivers)

        base = os.path.join(args.output_dir, f"{root}_{i:05d}")
        if args.format == "segy":
            write_segy_shot(base + ".segy", {"src": src, "rcv": rcv}, gather.shape[0],
                            max(1, int(round(dt_rec * 1e6))),
                            w_sx=args.word_sx, w_sy=args.word_sy, w_sz=args.word_sz,
                            w_rx=args.word_rx, w_ry=args.word_ry, w_rz=args.word_rz,
                            data=gather, field_record=i)
            print(f"  -> {base}.segy ({gather.shape[0]} samples x {gather.shape[1]} traces)")
        else:
            import h5py
            with h5py.File(base + ".h5", "w") as f:
                f.create_dataset("pressure", data=gather)
                f.create_dataset("source", data=np.asarray(src))
                f.create_dataset("receivers", data=np.asarray(rcv))
                f.attrs.update(dt=dt_rec, f0=args.f0, order=args.order, dim=dim, vp_file=args.vp_file)
            print(f"  -> {base}.h5 ({gather.shape[0]} samples x {gather.shape[1]} traces)")
        if args.plot:
            plot_shot(base + ".png", model_section(solver, mesh, dim), snapshot, snap_step * dt,
                      gather, dt_rec, src, rcv, lx, lz)
            print(f"  -> {base}.png")


if __name__ == "__main__":
    main()
    kokkos.finalize()
