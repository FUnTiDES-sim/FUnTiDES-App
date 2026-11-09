"""
Marmousi example of makutu.mesh.seismesh: mesh adapted to the SEP model for a Ricker source of
8 Hz (order 4, 5 points per wavelength), then one shot on it and figures (mesh coloured by vp,
zoom on the 2:1 transition layers, pressure snapshots, shot gather).

  python apps/mesh/examples/seismesh_marmousi.py --sep-dir ../../FUnTiDES-data/SEP/marmousi --out marmousi_mesh
"""
import argparse

import numpy as np
import kokkos
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sep-dir", required=True, help="directory of Vp_xz.H and Rho_xz.H")
parser.add_argument("--out", default="marmousi_mesh")
args = parser.parse_args()

from makutu.mesh.seismesh import build_mesh_from_sep  # noqa: E402
build_mesh_from_sep(f"{args.sep_dir}/Vp_xz.H", f0=8.0, out_dir=args.out, order=4, ppw=5,
                    rho_file=f"{args.sep_dir}/Rho_xz.H", lts=True)
kokkos.initialize()
import pyfuntides.solver as FTSolver  # noqa: E402
from makutu.mesh import GmshMesh  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402
S = D = args.out
m = GmshMesh(f"{D}/mesh.msh", 4, f"{D}/materials.csv")
vp, _ = m.materialOfElements()
C = m._corners[:, [0, 1, 3, 2], :].copy()            # vertices (0,0),(1,0),(1,1),(0,1) -> polygon order
C[:, :, 1] = m.top - C[:, :, 1]                       # depth
fig, ax = plt.subplots(figsize=(12, 4.6))
pc = PolyCollection(C / 1000, array=vp, cmap="viridis", edgecolors="k", linewidths=0.15)
ax.add_collection(pc); ax.set_xlim(0, 9.98); ax.set_ylim(3.46, 0); ax.set_aspect("equal")
ax.set_xlabel("x (km)"); ax.set_ylabel("profondeur (km)")
ax.set_title(f"Marmousi, maillage seismesh (f0 = 8 Hz, ordre 4, 5 pts/λ) : {m.n_elements} quadrangles\ndoublement du pas horizontal à 2,26 km, re-raffinement à 2,84 km", fontsize=10)
fig.colorbar(pc, ax=ax, label="vp (m/s)", shrink=0.8); fig.tight_layout(); fig.savefig(f"{S}/marmousi_mesh.png", dpi=150)
ax.set_xlim(3.5, 6.5); ax.set_ylim(3.46, 1.6); ax.set_title("Zoom : couches de transition 2:1 (doublement puis re-raffinement sous la zone lente)")
fig.savefig(f"{S}/marmousi_mesh_zoom.png", dpi=150)
# shot with snapshots
s = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT, dt=5e-4, minTime=0.0,
                   maxTime=3.0, dtSeismo=2e-3, sourceType="ricker2", sourceFreq=8.0, **m.solver_kwargs())
s.initialize(); rx = np.arange(100, 9900, 25.0)
s.updateSourceAndReceiversFromCoords([[4600.0, 0, 20.0]], [[x, 0, 20.0] for x in rx]); s.evaluateSource()
n = m.model.get_number_of_nodes()
xz = np.array([[m.model.node_coord(i, 0), m.top - m.model.node_coord(i, 2)] for i in range(n)])
snaps = {}
for i in range(s.n_time_steps):
    s.execute(i)
    if i in (1600, 2800): snaps[i * 5e-4] = np.array(s.data.get_current_field(0), copy=True)
d = s.getPressureAtReceivers()
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
for ax, (t, f) in zip(axs, snaps.items()):
    a = np.percentile(np.abs(f), 99.5)
    tc = ax.tripcolor(xz[:, 0] / 1000, xz[:, 1] / 1000, f, cmap="seismic", vmin=-a, vmax=a, shading="gouraud")
    ax.set_ylim(3.46, 0); ax.set_aspect("equal"); ax.set_title(f"Pression à t = {t:.1f} s (source x = 4,6 km)")
    ax.set_ylabel("profondeur (km)")
axs[-1].set_xlabel("x (km)"); fig.tight_layout(); fig.savefig(f"{S}/marmousi_snapshots.png", dpi=130)
fig, ax = plt.subplots(figsize=(10, 6))
a = np.percentile(np.abs(d), 98)
ax.imshow(d, aspect="auto", cmap="gray", vmin=-a, vmax=a, extent=[rx[0] / 1000, rx[-1] / 1000, 3.0, 0])
ax.set_xlabel("x récepteur (km)"); ax.set_ylabel("t (s)"); ax.set_title("Tir sur le maillage seismesh de Marmousi (f0 = 8 Hz, récepteurs à 20 m)")
fig.tight_layout(); fig.savefig(f"{S}/marmousi_shot.png", dpi=130)
del s, m; import gc; gc.collect(); kokkos.finalize()
