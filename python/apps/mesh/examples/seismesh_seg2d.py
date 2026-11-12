"""SEG/EAGE 2-D salt model: seismesh quadtree mesh (from the fastest velocity, 2:1 refinement), figures of the
element size and shot against the Cartesian makutu mesh.

  python seismesh_seg2d.py <output dir> <points per wavelength> <model .npz>
  (model: c_true of shape (nx, nz) sampled every 20 m, e.g. QSI-Examples/seg2d/model/model_256x128.npz)
"""
import sys, json, numpy as np, kokkos, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from makutu.input.createSepModel import _write_sep
from makutu.mesh.seismesh import build_mesh_from_sep, grid_from_sep
from makutu.mesh.seismesh.sizing import SizingParams, h_required
S = sys.argv[1]; F0, PPW, ORDER, T = 5.0, int(sys.argv[2]), 4, 3.0
m = np.load(sys.argv[3], allow_pickle=True)
c = m["c_true"].astype(np.float32)                                   # (nx, nz), 20 m
c10 = c                                                                  # 20 m cells
_write_sep(f"{S}/vp10.H", c10.T[None], 20.0, 20.0, 1.0)
r = build_mesh_from_sep(f"{S}/vp10.H", F0, f"{S}/mesh", order=ORDER, ppw=PPW, strategies=("quadtree",))
sel = r["selected"]; print(json.dumps({k: sel[k] for k in ("n_elements", "dt_cfl", "params")}, default=str))
# --- mesh figures
g = grid_from_sep(f"{S}/vp10.H"); h = h_required(g, SizingParams(2.5 * F0, ORDER, PPW))[:, 0, :]
kokkos.initialize()
import pyfuntides.solver as FTSolver
from makutu.mesh import GmshMesh, CartesianMesh2D
from makutu.solvers import AcousticSolver
mesh = GmshMesh(r["msh"], ORDER, r["materials"])
P = mesh._corners[:, [0, 1, 3, 2], :].copy(); P[:, :, 1] = mesh.top - P[:, :, 1]
size = np.array([np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1).max() for Q in P])
def hreq(Q):
    lo, hi = Q.min(0), Q.max(0)
    i0, i1 = int(np.floor(lo[0] / 20)), int(np.ceil(hi[0] / 20) - 1)
    k0, k1 = int(np.floor(lo[1] / 20)), int(np.ceil(hi[1] / 20) - 1)
    return h[i0:max(i1, i0) + 1, k0:max(k1, k0) + 1].min()
ratio = size / np.array([hreq(Q) for Q in P])
fig, axs = plt.subplots(3, 1, figsize=(12, 13))
im = axs[0].imshow(c10.T, extent=[0, 5120, 2560, 0], cmap="viridis")
axs[0].add_collection(LineCollection([np.vstack([Q, Q[:1]]) for Q in P], colors="w", linewidths=0.25))
fig.colorbar(im, ax=axs[0], label="vp (m/s)", shrink=0.8); axs[0].set_title("SEG/EAGE 2-D sel et maillage seismesh quadtree")
for ax, vals, cmap, lab, clim in [(axs[1], size, "viridis", "taille de l'élément (m)", None),
                                  (axs[2], ratio, "RdYlGn_r", "taille / taille requise (≤ 1 : 8 pts/λ)", (0, 1.2))]:
    pc = PolyCollection(P, array=vals, cmap=cmap, edgecolors="k", linewidths=0.15)
    if clim: pc.set_clim(*clim)
    ax.add_collection(pc); ax.set_xlim(0, 5120); ax.set_ylim(2560, 0); ax.set_aspect("equal"); fig.colorbar(pc, ax=ax, label=lab, shrink=0.8)
for ax in axs: ax.set_ylabel("profondeur (m)")
axs[2].set_xlabel("x (m)")
fig.suptitle(f"{sel['n_elements']} quadrangles, f0 = {F0:g} Hz, ordre {ORDER}, {PPW} pts/λ, jacobien min {sel['scaled_jacobian']['min']:.2f}, "
             f"éléments par niveau {sel['params']['elements_per_level']}", fontsize=10)
fig.savefig(f"{S}/seg2d_mesh.png", dpi=120, bbox_inches="tight")
# --- shots
src = [2560.0, 0.0, 40.0]; rx = np.arange(100.0, 5100.0, 50.0)
rcv = [[x, 0.0, 20.0] for x in rx]
def shot(kw, dt):
    s = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT, dt=dt, minTime=0.0,
                       maxTime=T, dtSeismo=dt, sourceType="ricker2", sourceFreq=F0, **kw)
    s.initialize(); s.updateSourceAndReceiversFromCoords([src], rcv); s.evaluateSource()
    for t in range(s.n_time_steps): s.execute(t)
    return s.getPressureAtReceivers().astype(float)
dt = 2e-4
d = shot(mesh.solver_kwargs(), dt)
ref_mesh = CartesianMesh2D(5120.0, 2560.0, 256, 128, order=ORDER, vp_file=f"{S}/vp10.H")
ref = shot(ref_mesh.solver_kwargs(), dt)
diff = np.linalg.norm(d - ref, axis=0) / np.linalg.norm(ref, axis=0)
glob = np.linalg.norm(d - ref) / np.linalg.norm(ref)
print("relative difference: global %.3f, per trace median %.3f, max %.3f" % (glob, np.median(diff), diff.max()))
np.savez(f"{S}/shots.npz", d=d, ref=ref, dt=dt)
fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
a = np.percentile(np.abs(ref), 98)
for ax, x, title in zip(axs, (ref, d, d - ref), ("cartésien (h = 20 m)", "seismesh quadtree", f"différence (globale {100 * glob:.1f} %)")):
    ax.imshow(x, aspect="auto", cmap="gray", vmin=-a, vmax=a, extent=[rx[0], rx[-1], T, 0]); ax.set_title(title); ax.set_xlabel("x (m)")
axs[0].set_ylabel("t (s)")
fig.suptitle("SEG/EAGE 2-D sel : tir (source x = 2560 m, z = 40 m ; récepteurs à 20 m), même échelle", fontsize=11)
fig.savefig(f"{S}/seg2d_shots.png", dpi=110, bbox_inches="tight")
del mesh, ref_mesh; import gc; gc.collect(); kokkos.finalize()
