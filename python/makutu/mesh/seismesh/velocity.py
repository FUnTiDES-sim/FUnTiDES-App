"""Chargement et interrogation d'une grille de vitesses différences finies.

Convention : tableau numpy indexé (ix, iy, iz), z = profondeur positive vers le bas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VelocityGrid:
    vel: np.ndarray            # (nx, ny, nz) en m/s (Vp ou Vs selon l'usage)
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    name: str = "vp"
    cell_centred: bool = False  # True : l'échantillon représente la cellule de largeur d centrée sur lui

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path, shape=None, spacing=None, origin=(0.0, 0.0, 0.0),
             dtype="float32", order="C", fortran_zfast=False) -> "VelocityGrid":
        """Charge .npy / .npz / binaire brut (+ fichier .json de métadonnées optionnel).

        Binaire brut : `shape=(nx,ny,nz)`, `spacing=(dx,dy,dz)` requis, sauf si un
        fichier `<path>.json` contient {"shape":..., "spacing":..., "origin":..., "dtype":...}.
        `fortran_zfast=True` pour les cubes stockés z-rapide (habituel en sismique) :
        le tableau est lu (nx,ny,nz) en ordre C avec z le plus rapide, ce qui est déjà
        notre convention ; utiliser order="F" si x est l'indice rapide.
        """
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if path.suffix == ".npz":
            d = np.load(path)
            return cls(d["vel"].astype(np.float32), tuple(d["origin"]), tuple(d["spacing"]))
        if path.suffix == ".npy":
            vel = np.load(path).astype(np.float32)
        else:
            shape = tuple(meta.get("shape", shape))
            dtype = meta.get("dtype", dtype)
            vel = np.fromfile(path, dtype=dtype).reshape(shape, order=meta.get("order", order))
            vel = vel.astype(np.float32)
        spacing = tuple(meta.get("spacing", spacing))
        origin = tuple(meta.get("origin", origin))
        if spacing is None or None in spacing:
            raise ValueError("spacing (dx,dy,dz) requis")
        return cls(vel, origin, spacing)

    def save_npz(self, path):
        np.savez_compressed(path, vel=self.vel, origin=np.array(self.origin),
                            spacing=np.array(self.spacing))

    # --------------------------------------------------------- géométrie
    @property
    def shape(self):
        return self.vel.shape

    @property
    def extent(self):
        """((xmin,xmax),(ymin,ymax),(zmin,zmax)) — bornes des nœuds de la grille, ou des cellules
        (``cell_centred``, demi-pas au-delà des échantillons extrêmes ; axe d'un seul échantillon
        inchangé)."""
        half = [0.5 * d if self.cell_centred and n > 1 else 0.0 for d, n in zip(self.spacing, self.shape)]
        return tuple((o - e, o + d * (n - 1) + e) for o, d, n, e in zip(self.origin, self.spacing, self.shape, half))

    def edge_axis(self, k):
        """Positions des échantillons de l'axe k, les deux extrêmes ramenés aux bornes du domaine
        (segmentation des axes : les segments couvrent tout le domaine)."""
        t = self.axis(k).astype(float)
        t[0], t[-1] = self.extent[k]
        return t

    def axis(self, k):
        return self.origin[k] + self.spacing[k] * np.arange(self.shape[k])

    def summary(self) -> dict:
        v = self.vel
        return {
            "shape": list(self.shape), "spacing": list(self.spacing), "origin": list(self.origin),
            "extent": [list(e) for e in self.extent],
            "vmin": float(v.min()), "vmax": float(v.max()), "vmean": float(v.mean()),
            "vmin_per_depth_quartile": [float(q.min()) for q in np.array_split(v, 4, axis=2)],
            "lateral_variability_pct": float(100 * np.mean(v.std(axis=(0, 1)) / v.mean(axis=(0, 1)))),
        }

    # ----------------------------------------------------- échantillonnage
    def sample(self, pts: np.ndarray) -> np.ndarray:
        """Interpolation trilinéaire aux points (N,3), coordonnées physiques, clamp aux bords."""
        idx = (pts - np.asarray(self.origin)) / np.asarray(self.spacing)
        n = np.asarray(self.shape)
        idx = np.clip(idx, 0, n - 1 - 1e-9)
        i0 = np.floor(idx).astype(int)
        f = idx - i0
        i1 = np.minimum(i0 + 1, n - 1)
        v = self.vel
        out = np.zeros(len(pts), dtype=np.float64)
        for cx, wx in ((i0[:, 0], 1 - f[:, 0]), (i1[:, 0], f[:, 0])):
            for cy, wy in ((i0[:, 1], 1 - f[:, 1]), (i1[:, 1], f[:, 1])):
                for cz, wz in ((i0[:, 2], 1 - f[:, 2]), (i1[:, 2], f[:, 2])):
                    out += wx * wy * wz * v[cx, cy, cz]
        return out

    def box_index(self, lo: np.ndarray, hi: np.ndarray):
        """Indices (premier, dernier) des échantillons d'une boîte (N, 3) : nœuds contenus, ou pour
        une grille ``cell_centred`` cellules recouvertes (même partiellement)."""
        o, d, n = np.asarray(self.origin), np.asarray(self.spacing), np.asarray(self.shape)
        lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
        if self.cell_centred:
            a = np.floor((lo - (o - 0.5 * d)) / d + 1e-6).astype(int)
            b = np.ceil((hi - (o - 0.5 * d)) / d - 1e-6).astype(int) - 1
        else:
            a = np.ceil((lo - o) / d - 1e-6).astype(int)
            b = np.floor((hi - o) / d + 1e-6).astype(int)
        a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
        return a, np.maximum(a, b)

    def box_mean(self, lo: np.ndarray, hi: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        """Moyenne de ``values`` (défaut : la vitesse) sur des boîtes (N, 3) : pondérée par la part
        de chaque cellule recouverte (grille ``cell_centred``), sinon moyenne des nœuds contenus."""
        v = self.vel if values is None else values
        a, b = self.box_index(lo, hi)
        lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
        out = np.empty(len(lo))
        o, d = np.asarray(self.origin), np.asarray(self.spacing)
        for e in range(len(lo)):
            blk = v[a[e, 0]:b[e, 0] + 1, a[e, 1]:b[e, 1] + 1, a[e, 2]:b[e, 2] + 1]
            if not self.cell_centred:
                out[e] = blk.mean()
                continue
            w = []
            for k in range(3):
                if self.shape[k] == 1:
                    w.append(np.ones(1))
                    continue
                idx = np.arange(a[e, k], b[e, k] + 1)
                c0 = o[k] + (idx - 0.5) * d[k]
                w.append(np.clip(np.minimum(c0 + d[k], hi[e, k]) - np.maximum(c0, lo[e, k]), 0.0, None))
            W = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
            out[e] = (W * blk).sum() / W.sum()
        return out

    def box_minmax(self, lo: np.ndarray, hi: np.ndarray):
        """(vmin, vmax, vmean) sur les nœuds de grille contenus dans des boîtes alignées (N,3).

        Le noyau DF étant échantillonné aux nœuds, on prend les nœuds dans la boîte
        (au moins le plus proche si la boîte est plus petite qu'une maille).
        Pour des hexaèdres non alignés, on passe leur bbox (conservatif).
        """
        a, b = self.box_index(lo, hi)
        vmin = np.empty(len(lo)); vmax = np.empty(len(lo)); vmean = np.empty(len(lo))
        v = self.vel
        for e in range(len(lo)):
            blk = v[a[e, 0]:b[e, 0] + 1, a[e, 1]:b[e, 1] + 1, a[e, 2]:b[e, 2] + 1]
            vmin[e], vmax[e], vmean[e] = blk.min(), blk.max(), blk.mean()
        return vmin, vmax, vmean


def synthetic_model(nx=201, ny=161, nz=121, d=25.0, seed=0) -> VelocityGrid:
    """Modèle test : gradient + couches ondulées + corps de sel + zone lente proche surface."""
    rng = np.random.default_rng(seed)
    x = np.arange(nx) * d
    y = np.arange(ny) * d
    z = np.arange(nz) * d
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Lx, Ly, Lz = x[-1], y[-1], z[-1]
    v = 1600.0 + 0.6 * Z                                    # gradient de compaction
    for k, (zc, dv) in enumerate([(0.25, 250), (0.45, 400), (0.7, 600)]):
        ondul = 0.04 * Lz * np.sin(2 * np.pi * (X / Lx + 0.3 * k)) * np.cos(np.pi * Y / Ly)
        v += dv * (Z > zc * Lz + ondul)
    salt = ((X - 0.6 * Lx) / (0.18 * Lx)) ** 2 + ((Y - 0.5 * Ly) / (0.2 * Ly)) ** 2 + \
           ((Z - 0.55 * Lz) / (0.15 * Lz)) ** 2 < 1
    v[salt] = 4500.0
    weather = (Z < 0.06 * Lz) & (((X - 0.25 * Lx) ** 2 + (Y - 0.3 * Ly) ** 2) < (0.2 * Lx) ** 2)
    v[weather] = 1200.0                                      # chenal lent proche surface
    v += rng.normal(0, 15, v.shape)
    return VelocityGrid(v.astype(np.float32), (0.0, 0.0, 0.0), (d, d, d), name="vp_synth")
