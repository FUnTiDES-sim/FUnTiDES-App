"""Gabarits de 3-raffinement de Tong & Zhang (2026), traduits dans makutu.mesh.seismesh.templates3_es.

Référence : H. Tong, Y. J. Zhang, « Element-saving hexahedral 3-refinement templates », Computer Aided
Geometric Design (2026) 102550, doi:10.1016/j.cagd.2026.102550 ; code d'origine CMU-CBML (licence MIT),
https://github.com/CMU-CBML/Element-Saving-Hexahedral-3-Refinement-Templates.

Contrôles : jacobien positif et volume exact pour les 256 codes ; conformité (chaque face intérieure
partagée par exactement deux hexaèdres, les autres sur le bord) sur des motifs raffinés aléatoires et
sur une boîte raffinée ; figures des 15 classes et d'une coupe du maillage à deux niveaux.
"""
import numpy as np
import pytest

from makutu.mesh.seismesh import templates3_es as T

_CORNER = np.array([(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
                    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)])
_FACES = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])


def scaled_jacobian(X):
    """Jacobien normalisé minimal aux 8 coins de chaque hexaèdre X (ne, 8, 3)."""
    c = X[:, _CORNER]
    e1, e2, e3 = (c[:, :, k] - c[:, :, 0] for k in (1, 2, 3))
    det = np.einsum("eij,eij->ei", np.cross(e1, e2), e3)
    return (det / (np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2) * np.linalg.norm(e3, axis=2))).min(1)


def hex_volume(X):
    """Volume exact d'hexaèdres trilinéaires (5 tétraèdres ne suffisent pas : 24 tétraèdres par faces)."""
    centre = X.mean(1)
    vol = np.zeros(len(X))
    for f in _FACES:
        fc = X[:, f].mean(1)
        for a, b in zip(f, np.roll(f, -1)):
            vol += np.einsum("ij,ij->i", np.cross(X[:, a] - fc, X[:, b] - fc), centre - fc) / 6.0
    return -vol


def face_census(nodes, hexes, lo, hi):
    """(faces partagées par plus de deux hexaèdres, faces isolées hors du bord de [lo, hi])."""
    count = {}
    for h in hexes:
        for f in _FACES:
            key = tuple(sorted(h[f]))
            count[key] = count.get(key, 0) + 1
    over = [k for k, n in count.items() if n > 2]
    tol = 1e-9 * np.max(np.asarray(hi) - np.asarray(lo))
    loose = []
    for k, n in count.items():
        if n == 1:
            P = nodes[list(k)]
            if not any(np.all(np.abs(P[:, d] - lo[d]) < tol) or np.all(np.abs(P[:, d] - hi[d]) < tol) for d in range(3)):
                loose.append(k)
    return over, loose


def test_every_code_valid():
    """Les 256 codes : hexaèdres non dégénérés, qui remplissent exactement la cellule unité."""
    worst = {}
    for code in range(256):
        X = np.array(T.cell_hexes((0, 0, 0), (1, 1, 1), code))
        sj = scaled_jacobian(X)
        assert sj.min() > 0.05, (code, sj.min())
        assert abs(hex_volume(X).sum() - 1.0) < 1e-9, code
        cls = T.ROTATIONS[code][0]
        worst[cls] = min(worst.get(cls, 1.0), sj.min())
    assert sorted(worst) == list(range(1, 16))


def _check_two_level(refined, h=(30.0, 30.0, 30.0)):
    nodes, hexes, _ = T.two_level_mesh((0, 0, 0), h, refined)
    X = nodes[hexes]
    sj = scaled_jacobian(X)
    vol = hex_volume(X)
    hi = np.array(h) * refined.shape
    assert sj.min() > 0.05, sj.min()
    assert abs(vol.sum() - np.prod(hi)) < 1e-6 * np.prod(hi)
    over, loose = face_census(nodes, hexes, (0, 0, 0), hi)
    assert not over and not loose, (len(over), len(loose))
    return nodes, hexes, sj


@pytest.mark.parametrize("seed", range(6))
def test_random_patterns_conforming(seed):
    """Cellules raffinées tirées au hasard (densités variées) : maillage conforme et valide."""
    rng = np.random.default_rng(seed)
    refined = rng.random((5, 5, 4)) < (0.1 + 0.08 * seed)
    _check_two_level(refined)


def test_box_conforming(figure_path):
    """Boîte de cellules raffinées au centre d'une grille grossière : le cas du sel encadré."""
    refined = np.zeros((8, 6, 6), bool)
    refined[2:5, 2:4, 1:4] = True
    nodes, hexes, sj = _check_two_level(refined)
    path = figure_path("coupe")
    if path is not None:
        from makutu.mesh.seismesh.plots import hex_section, _save
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, y in zip(axs, (75.0, 31.0)):
            polys, ids = hex_section(nodes, hexes, y)
            pc = PolyCollection(polys, array=sj[ids], cmap="viridis", edgecolors="k", linewidths=0.4)
            pc.set_clim(0, 1)
            ax.add_collection(pc)
            ax.set_xlim(0, 240)
            ax.set_ylim(180, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            ax.set_title(f"coupe y = {y:.0f} m ({'dans' if y == 75 else 'hors de'} la boîte raffinée)")
            fig.colorbar(pc, ax=ax, shrink=0.8, label="jacobien normalisé")
        fig.suptitle(f"Gabarits Tong & Zhang (2026) : boîte 3×2×3 raffinée dans une grille 8×6×6 de 30 m — "
                     f"{len(hexes)} hexaèdres, jacobien min {sj.min():.2f}, conforme", fontsize=10)
        _save(fig, path)


def test_class_gallery(figure_path):
    """Figure : un exemple de chaque classe (sommets noirs en rouge), hexaèdres légèrement écartés."""
    path = figure_path()
    if path is None:
        pytest.skip("figures désactivées")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from makutu.mesh.seismesh.plots import _save
    fig = plt.figure(figsize=(20, 12))
    for cls in range(1, 16):
        code = min(k for k, v in T.ROTATIONS.items() if v[0] == cls)
        X = np.array(T.cell_hexes((0, 0, 0), (1, 1, 1), code))
        sj = scaled_jacobian(X)
        ax = fig.add_subplot(3, 5, cls, projection="3d")
        c = X.mean(1)
        Xe = X + 0.25 * (c - 0.5)[:, None, :]                          # exploded view
        polys = [Xe[e][list(f)] for e in range(len(X)) for f in _FACES]
        cols = plt.cm.viridis(np.repeat(sj, 6))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.3, alpha=0.55))
        black = [i for i in range(8) if code & (128 >> i)]
        P = T._CORNERS[black] * 1.25 - 0.125
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="r", s=40)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_zlim(-0.15, 1.15)
        ax.set_axis_off()
        ax.set_title(f"classe {cls}, code {code:08b}\n{len(X)} hexaèdres, jacobien min {sj.min():.2f}", fontsize=9)
    fig.suptitle("Gabarits de 3-raffinement de Tong & Zhang (2026) — un exemple par classe ; points rouges = "
                 "sommets touchés par une cellule fine ; couleur = jacobien normalisé (viridis, 0 → 1)", fontsize=11)
    _save(fig, path)


@pytest.mark.parametrize("n", [3, 4])
def test_coarse_island_conforming(n, figure_path):
    """Cas du sel : îlot n×n×n de cellules grossières dans un domaine raffiné. Les cellules de bord de
    l'îlot portent les transitions, l'intérieur reste grossier et régulier."""
    refined = np.ones((n + 4, n + 4, n + 4), bool)
    refined[2:2 + n, 2:2 + n, 2:2 + n] = False
    nodes, hexes, sj = _check_two_level(refined)
    _, _, cls = T.two_level_mesh((0, 0, 0), (30.0, 30.0, 30.0), refined)
    n_coarse = int(np.sum(cls == 1))
    assert n_coarse >= (n - 2) ** 3                       # interior of the island untouched
    path = figure_path()
    if path is not None:
        from makutu.mesh.seismesh.plots import hex_section, _save
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        L = 30.0 * (n + 4)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        polys, ids = hex_section(nodes, hexes, L / 2 + 1.0)
        pc = PolyCollection(polys, array=sj[ids], cmap="viridis", edgecolors="k", linewidths=0.3)
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        ax.set_xlim(0, L)
        ax.set_ylim(L, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(pc, ax=ax, shrink=0.8, label="jacobien normalisé")
        ax.set_title(f"Îlot grossier {n}×{n}×{n} (cellules de 30 m) dans un domaine fin (10 m)\n"
                     f"coupe au milieu en y — {len(hexes)} hexaèdres, jacobien min {sj.min():.2f}, "
                     f"{n_coarse} cellules grossières intactes", fontsize=9)
        _save(fig, path)
