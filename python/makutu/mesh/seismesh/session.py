"""Session de maillage : modèle, critère de taille, candidats, recherche déterministe (autopilote), export."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .velocity import VelocityGrid, synthetic_model
from .sizing import SizingParams, h_required, ideal_element_count, segment_all_axes, segment_axis, axis_profile
from .mesher import mesh_tensor, mesh_extrude, mesh_doubling, mesh_hybrid, mesh_refine3, mesh_tetsplit
from .quality import evaluate, public


class Session:
    def __init__(self, workdir="seismesh_out", lts: bool = False, lts_max_level: int = 10):
        self.lts = lts                      # le solveur utilise-t-il un pas de temps local ?
        self.lts_max_level = lts_max_level
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.grid: VelocityGrid | None = None
        self.sizing = SizingParams()
        self.h = None
        self.candidates: dict[str, dict] = {}
        self._n = 0
        self.log: list[dict] = []

    # ------------------------------------------------------------ outils
    def load_model(self, path: str | None = None, synthetic: bool = False, shape=None,
                   spacing=None, origin=(0, 0, 0), dtype="float32", order="C"):
        if synthetic or path is None:
            self.grid = synthetic_model()
        else:
            self.grid = VelocityGrid.load(path, shape=shape, spacing=spacing, origin=origin,
                                          dtype=dtype, order=order)
        self.h = None
        return self.grid.summary()

    def describe_model(self):
        self._need_grid()
        s = self.grid.summary()
        z = self.grid.axis(2)
        vz = self.grid.vel.min(axis=(0, 1))
        s["vmin_depth_profile"] = [[float(z[i]), float(vz[i])] for i in np.linspace(0, len(z) - 1, 12).astype(int)]
        return s

    def set_sizing(self, fmax: float, order: int = 4, ppw: float = 5.0, safety: float = 1.0,
                   hmin: float = 0.0, hmax: float | None = None):
        self._need_grid()
        self.sizing = SizingParams(fmax, order, ppw, safety, hmin, np.inf if hmax is None else hmax)
        self.h = h_required(self.grid, self.sizing)
        lam = self.grid.vel.min() / fmax
        return {"sizing": self.sizing.to_dict(), "lambda_min": float(lam),
                "h_required": {"min": float(self.h.min()), "max": float(self.h.max()),
                               "median": float(np.median(self.h))},
                "fd_grid_spacing": list(self.grid.spacing),
                "warning": ("h_req min < pas DF : le modèle est sous-résolu par rapport au critère "
                            "(on ne peut pas mailler plus fin que l'information disponible)")
                if self.h.min() < min(self.grid.spacing) else None,
                "ideal_elements_lower_bound": int(ideal_element_count(self.grid, self.h))}

    def plan_segments(self, max_segments=(6, 6, 8), max_grading=2.0, min_elems_per_segment=2):
        self._need_h()
        segs = segment_all_axes(self.grid, self.h, tuple(max_segments), max_grading=max_grading,
                                min_elems_per_segment=min_elems_per_segment)
        n = [sum(s.n for s in a) for a in segs]
        return {"segments": [[{"t0": s.t0, "t1": s.t1, "n": s.n, "spacing": round(s.spacing, 2)}
                              for s in a] for a in segs],
                "elements_per_axis": n, "predicted_tensor_elements": int(np.prod(n))}

    def generate_mesh(self, strategy="tensor", max_segments=(6, 6, 8), max_grading=2.0,
                      min_elems_per_segment=2, size_factor=0.8, quad_algo="blossom",
                      max_levels=3, trans_layers=1, allow_refine=True, q_bg=0.1, lmax3=1,
                      cfl=0.5, t_sim=4.0):
        self._need_h()
        t0 = time.time()
        h_seg = self.h
        if strategy == "hybrid":
            # base grid sized for the background (lateral quantile q_bg of h_req at each depth): the
            # localized slow anomalies are then handled by the local 3-refinement, not by the grid
            h_seg = np.ascontiguousarray(np.broadcast_to(
                np.quantile(self.h, q_bg, axis=(0, 1))[None, None, :], self.h.shape))
        segs = segment_all_axes(self.grid, h_seg, tuple(max_segments), max_grading=max_grading,
                                min_elems_per_segment=min_elems_per_segment)
        self._n += 1
        cid = f"c{self._n:03d}"
        msh = str(self.workdir / f"{cid}_{strategy}.msh")
        if strategy == "tensor":
            m = mesh_tensor(self.grid, segs, out_msh=msh)
        elif strategy == "extrude":
            m = mesh_extrude(self.grid, self.h, segs[2], quad_algo=quad_algo,
                             size_factor=size_factor, out_msh=msh)
        elif strategy == "doubling":
            m = mesh_doubling(self.grid, self.h, segs, max_levels=max_levels,
                              trans_layers=trans_layers, allow_refine=allow_refine, out_msh=msh)
        elif strategy == "hybrid":
            m = mesh_hybrid(self.grid, self.h, segs, max_levels=max_levels, q_bg=q_bg,
                            lmax3=lmax3, out_msh=msh)
        elif strategy == "refine3":
            # from the fastest velocity: base grid for size_factor * largest required size, 1:3 refinement
            m = mesh_refine3(self.grid, self.h, base_max_segments=tuple(max_segments), max_grading=max_grading,
                             out_msh=msh, base_size=size_factor * float(self.h.max()))
        elif strategy == "tetsplit":
            m = mesh_tetsplit(self.grid, self.h, size_factor=size_factor, out_msh=msh)
        else:
            raise ValueError("strategy ∈ {tensor, extrude, doubling, hybrid, refine3, tetsplit}")
        rep = evaluate(m, self.grid, self.sizing, cfl=cfl, t_sim=t_sim,
                       lts_max_level=self.lts_max_level)
        rep["mesh_time_s"] = round(time.time() - t0, 2)
        rep["inputs"] = {"strategy": strategy, "max_segments": list(max_segments),
                         "max_grading": max_grading, "size_factor": size_factor,
                         "quad_algo": quad_algo, "max_levels": max_levels,
                         "trans_layers": trans_layers, "allow_refine": allow_refine, "q_bg": q_bg, "lmax3": lmax3, "sizing": self.sizing.to_dict()}
        if strategy == "refine3":
            rep["refine3"] = {k: m.params[k] for k in ("lmax", "base_size", "base_cells", "templates", "leaves_per_level")
                              if k in m.params}
        if strategy in ("doubling", "hybrid"):
            rep["doubling"] = {k: m.params[k] for k in ("transitions", "nx_ny_fine", "nz_layers")}
        if strategy == "hybrid":
            rep["refine3"] = {"zones": [{k: z[k] for k in ("z", "level", "templates")}
                                        for z in m.params["refined_zones"]],
                              "blocked_layers": m.params["blocked_layers"]}
        self.candidates[cid] = {"mesh": m, "report": rep}
        out = public(rep)
        out["candidate_id"] = cid
        return out

    def compare_candidates(self, lts: bool | None = None):
        lts = self.lts if lts is None else lts
        rows = []
        for cid, c in self.candidates.items():
            r = c["report"]
            rows.append({"id": cid, "ok": r["ok"], "strategy": r["strategy"],
                         "inputs": r["inputs"], "n_elements": r["n_elements"],
                         "sj_min": round(r["scaled_jacobian"]["min"], 3),
                         "ppw_min": round(r["ppw_effective"]["min"], 2),
                         "frac_under": r["frac_undersampled"], "dt": r["dt_cfl"],
                         "cost_index": round(r["cost_index"], 3),
                         "cost_index_lts": round(r["cost_index_lts"], 3),
                         "efficiency": round(r["efficiency_vs_ideal"], 3)})
        key = "cost_index_lts" if lts else "cost_index"
        rows.sort(key=lambda r: (not r["ok"], r[key]))
        return {"ranking": key, "candidates": rows,
                "best_ok": next((r["id"] for r in rows if r["ok"]), None)}

    def export_mesh(self, candidate_id: str, basename: str = "mesh"):
        c = self.candidates[candidate_id]
        m, rep = c["mesh"], c["report"]
        pe = rep["_per_element"]
        base = self.workdir / basename
        shutil.copy(m.msh_path, f"{base}.msh")
        np.savez_compressed(f"{base}_sem.npz", nodes=m.nodes, hexes=m.hexes,
                            vp_min=pe["vmin"], vp_max=pe["vmax"], vp_mean=pe["vmean"])
        write_vtk(f"{base}.vtk", m.nodes, m.hexes,
                  {"vp_mean": pe["vmean"], "vp_min": pe["vmin"], "ppw": pe["ppw"],
                   "scaled_jacobian": pe["sj"], "dt_cfl": pe["dt"]})
        Path(f"{base}_report.json").write_text(json.dumps(public(rep), indent=2, default=float))
        return {"files": [f"{base}.msh", f"{base}_sem.npz", f"{base}.vtk", f"{base}_report.json"]}

    def autopilot(self, strategies=("tensor", "extrude", "doubling", "hybrid"), t_sim=4.0, lts=None):
        """Recherche déterministe : minimise cost_index sous contraintes de qualité."""
        self._need_h()
        tried = []
        # balayage resserré (les variantes redondantes donnent des maillages identiques)
        if "tensor" in strategies:
            for ms in [(3, 3, 5), (6, 6, 10)]:
                tried.append(self.generate_mesh("tensor", ms, 2.0, t_sim=t_sim))
        if "doubling" in strategies:
            for mxy in (1, 3):
                for lev in (2, 3):
                    tried.append(self.generate_mesh("doubling", (mxy, mxy, 10), 2.0, max_levels=lev,
                                                    allow_refine=True, t_sim=t_sim))
        if "hybrid" in strategies:
            for mxy in (1, 3):
                for lev in (2, 3):
                    tried.append(self.generate_mesh("hybrid", (mxy, mxy, 10), 2.0,
                                                    max_levels=lev, t_sim=t_sim))
        if "refine3" in strategies:
            for sf in (1.0, 0.9, 0.8, 0.7):            # base size: fraction of the largest required size
                tried.append(self.generate_mesh("refine3", (1, 1, 10), 2.0, size_factor=sf, t_sim=t_sim))
        if "tetsplit" in strategies:
            for sf in (0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35):   # tetrahedron size: fraction of 2 h_req, until sampled
                r = self.generate_mesh("tetsplit", (6, 6, 8), 2.0, size_factor=sf, t_sim=t_sim)
                tried.append(r)
                if r["ok"]:
                    break
        if "extrude" in strategies:
            for sf in (0.8, 0.7, 0.6):
                r = self.generate_mesh("extrude", (6, 6, 8), 2.0, size_factor=sf, t_sim=t_sim)
                tried.append(r)
                if r["ok"]:
                    break
        return self.compare_candidates(lts)

    # ---------------------------------------------------------- interne
    def _need_grid(self):
        if self.grid is None:
            raise RuntimeError("aucun modèle chargé : appeler load_model")

    def _need_h(self):
        self._need_grid()
        if self.h is None:
            raise RuntimeError("critère non défini : appeler set_sizing")


def write_vtk(path, nodes, hexes, cell_data: dict):
    """VTK legacy binaire (ParaView). Ordre des nœuds hexa gmsh == VTK_HEXAHEDRON."""
    ne = len(hexes)
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nseismesh\nBINARY\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(nodes)} float\n".encode())
        f.write(nodes.astype(">f4").tobytes())
        f.write(f"\nCELLS {ne} {ne * 9}\n".encode())
        cells = np.hstack([np.full((ne, 1), 8), hexes]).astype(">i4")
        f.write(cells.tobytes())
        f.write(f"\nCELL_TYPES {ne}\n".encode())
        f.write(np.full(ne, 12, dtype=">i4").tobytes())
        f.write(f"\nCELL_DATA {ne}\n".encode())
        for k, v in cell_data.items():
            f.write(f"SCALARS {k} float 1\nLOOKUP_TABLE default\n".encode())
            f.write(np.asarray(v, dtype=">f4").tobytes())
            f.write(b"\n")
