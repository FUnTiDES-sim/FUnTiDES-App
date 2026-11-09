"""Shots on the seismesh meshes of a 3-layer model with a slow inclusion in the second layer, against
the Cartesian makutu mesh (interfaces on its element faces), and figures.

  python seismesh_inclusion_shots.py <dir> 2d|3d
<dir> holds vp2d.H / vp3d.H (3 layers 1800 / 3000 / 4000 m/s, interfaces at 500 and 1000 m, sphere
of 1500 m/s, radius 200 m, centred at (1000, 1000, 750) m; SEP every 10 m in 2-D, 12.5 m in 3-D)
and out2d / out3d from build_mesh_from_sep(f0=5, order 4 in 2-D / 3 in 3-D, ppw=8).
"""
import sys, json, numpy as np, kokkos, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
kokkos.initialize()
import pyfuntides.solver as FTSolver
from makutu.input.SEPReader import read_sep
from makutu.mesh import GmshMesh, CartesianMesh2D, StructuredMesh
from makutu.solvers import AcousticSolver
S, dim = sys.argv[1], sys.argv[2]
F0, T, DT = 5.0, 1.6, 4e-4
order = 4 if dim == "2d" else 3
rep = json.load(open(f"{S}/out{dim}/report.json"))["selected"]
dt = min(DT, 0.6 * rep["dt_cfl"])
src = [1000.0, 1000.0, 60.0]
xs = np.arange(100.0, 1950.0, 100.0)
rcv = [[x, 1000.0, 40.0] for x in xs] + [[1000.0, 1000.0, 1300.0], [1000.0, 1000.0, 1300.0]]
rcv[-1] = [600.0, 1000.0, 750.0]                       # beside the inclusion
kw = dict(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
          model_location=FTSolver.ModelLocationType.ONELEMENTS, dt=dt, minTime=0.0, maxTime=T, dtSeismo=dt,
          sourceType="ricker2", sourceFreq=F0)
def shot(mk, w=None, snap=None):
    s = AcousticSolver(**kw, **mk); s.initialize(); s.updateSourceAndReceiversFromCoords([src], rcv, sourceWeights=w)
    s.evaluateSource(); field = None
    for t in range(s.n_time_steps):
        s.execute(t)
        if snap is not None and t == snap: field = np.array(s.data.get_current_field(0), copy=True)
    return s.getPressureAtReceivers().astype(float), field
mesh = GmshMesh(f"{S}/out{dim}/mesh.msh", order, f"{S}/out{dim}/materials.csv"); print(mesh, "dt", dt)
snap = int(0.75 / dt)
d, f = shot(mesh.solver_kwargs(), snap=snap)
if dim == "2d":
    ref_mesh = CartesianMesh2D(2000.0, 1500.0, 100, 75, order=order, vp_file=f"{S}/vp2d.H")
    ref, _ = shot(ref_mesh.solver_kwargs())
else:
    ref_mesh = StructuredMesh(2000.0, 2000.0, 1500.0, 80, 80, 60, order=order, vp_file=f"{S}/vp3d.H")
    h = 25.0; p = np.array([src[0], src[1], 1500.0 - src[2]]); xi = 2 * (p - np.floor(p / h) * h) / h - 1
    ref, _ = shot(ref_mesh.solver_kwargs(), mesh.sourceWeights(xi)[None])
diff = np.linalg.norm(d - ref, axis=0) / np.linalg.norm(ref, axis=0)
lag = [(np.argmax(np.correlate(d[:, j], ref[:, j], "full")) - (len(ref) - 1)) * dt for j in range(d.shape[1])]
print("rel diff", np.round(diff, 3)); print("lag ms", np.round(1e3 * np.array(lag), 1))
np.savez(f"{S}/shots{dim}.npz", d=d, ref=ref, dt=dt, diff=diff)

# --- figures: mesh section over the model, snapshot, traces
sep = read_sep(f"{S}/vp{dim}.H")
vel = sep.data[0] if dim == "2d" else sep.data[:, sep.n2 // 2, :]          # (z, x) section y = 1000
ext = [0, 2000, 1500, 0]
C = mesh._corners
if dim == "2d":
    P = C[:, [0, 1, 3, 2], :].copy(); P[:, :, 1] = mesh.top - P[:, :, 1]
    segs = np.concatenate([P[:, [i, (i + 1) % 4]] for i in range(4)])
else:
    ys = np.unique(np.round(C[:, :, 1], 3)); y0 = ys[np.argmin(abs(ys - 1000.0))]
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]   # edges in x and z
    segs = []
    for e in range(len(C)):
        for a, b in E:
            if abs(C[e, a, 1] - y0) < 1e-6 and abs(C[e, b, 1] - y0) < 1e-6:
                segs.append([[C[e, a, 0], mesh.top - C[e, a, 2]], [C[e, b, 0], mesh.top - C[e, b, 2]]])
    segs = np.array(segs)
fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
ax = axs[0]
im = ax.imshow(vel, extent=ext, cmap="viridis", aspect="equal")
ax.add_collection(LineCollection(segs, colors="w", linewidths=0.35))
ax.set_title(f"Maillage seismesh ({rep['strategy']}, {mesh.n_elements} éléments)" + ("" if dim == "2d" else ", coupe y ≈ 1 km"))
ax.set_xlabel("x (m)"); ax.set_ylabel("profondeur (m)"); fig.colorbar(im, ax=ax, shrink=0.7, label="vp (m/s)")
ax = axs[1]
M = mesh.model; n = M.get_number_of_nodes()
X = np.array([[M.node_coord(i, 0), M.node_coord(i, 1), mesh.top - M.node_coord(i, 2)] for i in range(n)])
sel = np.ones(n, bool) if dim == "2d" else np.abs(X[:, 1] - y0) < 1e-6
a = np.percentile(np.abs(f[sel]), 99.5)
ax.tripcolor(X[sel, 0], X[sel, 2], f[sel], cmap="seismic", vmin=-a, vmax=a, shading="gouraud")
ax.contour(np.linspace(0, 2000, vel.shape[1]), np.linspace(0, 1500, vel.shape[0]), vel, levels=[1600, 2400, 3500], colors="k", linewidths=0.6)
ax.set_ylim(1500, 0); ax.set_aspect("equal"); ax.set_title(f"Pression à t = {snap * dt:.2f} s"); ax.set_xlabel("x (m)")
ax = axs[2]; t = np.arange(len(d)) * dt
sel_tr = [1, 5, 9, 13, 17, len(rcv) - 2, len(rcv) - 1]
for k, j in enumerate(sel_tr):
    sc = 0.45 / np.abs(ref[:, j]).max()
    ax.plot(t, k + ref[:, j] * sc, "k", lw=1.5, label="cartésien (interfaces suivies)" if k == 0 else None)
    ax.plot(t, k + d[:, j] * sc, "r--", lw=1.0, label="seismesh" if k == 0 else None)
    lab = f"x = {rcv[j][0]:.0f}, z = {rcv[j][2]:.0f}"
    ax.text(T, k + 0.22, f"{lab} : {100 * diff[j]:.1f} %, {1e3 * lag[j]:+.0f} ms", ha="right", fontsize=7.5)
ax.set_yticks([]); ax.set_xlabel("t (s)"); ax.legend(loc="lower left", fontsize=8); ax.set_title("Traces normalisées : écart relatif, décalage")
fig.suptitle(f"{dim.upper()} : 3 couches (1800 / 3000 / 4000 m/s) et inclusion lente 1500 m/s (rayon 200 m) dans la 2e couche ; f0 = 5 Hz, ordre {order}, 8 pts/λ", fontsize=11)
fig.savefig(f"{S}/inclusion_{dim}.png", dpi=125, bbox_inches="tight")
del mesh, ref_mesh, M; import gc; gc.collect(); kokkos.finalize()
