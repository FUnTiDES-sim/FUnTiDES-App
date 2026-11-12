"""Registre des modélisations à reproduire : modèle -> études (application, options, valeurs attendues).

Utilisé par ``reproduce.py``. Chaque étude lance une application de ``apps/mortar`` avec des options
fixées, dans ``<sortie>/<modèle>/<étude>/``. Quand ``expected`` est donné, ``reproduce.py`` relit le JSON
produit et compare :

- ``errors`` : {maillage: (écart traces, écart instantané)}, fractions (0,0313 = 3,13 %), tolérance
  absolue ``ERR_TOL`` ;
- ``elements`` : {maillage: nombre d'éléments}, égalité exacte (les maillages sont déterministes).

Les valeurs attendues viennent des runs du 12 novembre 2026 (FUnTiDES-App ``69643bb``, doc
``apps/mortar/doc/mortar-maths-et-implementation.md`` § 9). Les études GPU (FUnTiDES) n'ont pas de
valeurs attendues : leurs résultats publiés (README de ``makutu/mortar``) dépendent d'options à
revérifier.

Modèles : ``FUnTiDES-data/SEP/<modèle>/Vp_xz.H`` ; ``disque`` et ``onde-plane`` sont synthétiques.
"""
from dataclasses import dataclass, field

ERR_TOL = 0.002          # 0,2 point de pourcentage


@dataclass
class Study:
    app: str                                   # script de apps/mortar
    args: list                                 # options, sans --out
    json: str | None = None                    # fichier de résultats écrit dans --out
    figures: tuple = ()                        # figures écrites dans --out
    gpu: bool = False                          # calcul FUnTiDES sur GPU
    runtime: str = ""                          # durée indicative (portable, 1 cœur NumPy / RTX 4070)
    expected: dict = field(default_factory=dict)
    note: str = ""


CASES = {
    "seg2d": dict(
        description="SEG/EAGE 2-D salt (5,1 x 2,6 km, sel mince 300-400 m)",
        studies=dict(
            compare=Study(
                "hybrid_compare_2d.py", ["--model", "seg2d", "--f0", "8", "--tmax", "3"],
                json="compare_seg2d.json", figures=("compare_seg2d.png", "sismos_seg2d.png"), runtime="2 min",
                expected=dict(errors={"hybride": (0.0313, 0.1019), "CG + DG": (0.0273, 0.1034),
                                      "blocs (escalier)": (0.0472, 0.1290), "uniforme": (0.0364, 0.0987)},
                              elements={"hybride": 2745, "CG + DG": 3037, "blocs (escalier)": 2610,
                                        "uniforme": 3698, "référence": 13976})),
            cost=Study(
                "mesh_cost_2d.py", ["--model", "seg2d", "--f0", "8", "--tmax", "3", "--zoom", "1500", "3500", "600", "1400"],
                json="cout_seg2d.json", figures=("maillages_seg2d.png",), runtime="1 min",
                expected=dict(elements={"hybride mortar": 2745, "CG + DG": 3037, "blocs (escalier)": 2610,
                                        "uniforme": 3698}),
                note="temps par pas dépendants de la machine : non contrôlés"),
            shots_gpu=Study(
                "blocks_shots_2d.py", ["--model", "seg2d", "--f0", "8", "--tmax", "4"],
                json="seg2d.json", figures=("seg2d.png",), gpu=True, runtime="2 min"),
        )),
    "sigsbee2a": dict(
        description="Sigsbee 2A (24,4 x 9,2 km, sel épais)",
        studies=dict(
            compare=Study(
                "hybrid_compare_2d.py", ["--model", "sigsbee2a", "--f0", "4", "--tmax", "6", "--tri-factor", "1",
                                         "--cg-tri-factor", "1"],
                json="compare_sigsbee2a.json", figures=("compare_sigsbee2a.png", "sismos_sigsbee2a.png"),
                runtime="15 min",
                expected=dict(errors={"hybride": (0.0619, 0.0568), "CG + DG": (0.0673, 0.0584),
                                      "blocs (escalier)": (0.0875, 0.0769), "uniforme": (0.0735, 0.0667)},
                              elements={"hybride": 11967, "CG + DG": 12618, "blocs (escalier)": 10586,
                                        "uniforme": 16960, "référence": 59932})),
            compare_tri43=Study(
                "hybrid_compare_2d.py", ["--model", "sigsbee2a", "--f0", "4", "--tmax", "6"],
                json="compare_sigsbee2a.json", figures=("compare_sigsbee2a.png", "sismos_sigsbee2a.png"),
                runtime="15 min",
                expected=dict(errors={"hybride": (0.0941, 0.0852), "CG + DG": (0.0995, 0.0863),
                                      "blocs (escalier)": (0.0875, 0.0769), "uniforme": (0.0735, 0.0667)}),
                note="triangles 4/3 (défaut) : sous-résolus, montre l'effet de la taille des triangles"),
            cost=Study(
                "mesh_cost_2d.py", ["--model", "sigsbee2a", "--f0", "4", "--tmax", "6", "--tri-factor", "1",
                                    "--zoom", "15000", "21000", "2300", "6000"],
                json="cout_sigsbee2a.json", figures=("maillages_sigsbee2a.png",), runtime="5 min",
                expected=dict(elements={"hybride mortar": 11967, "CG + DG": 12618, "blocs (escalier)": 10586,
                                        "uniforme": 16960})),
            shots_gpu=Study(
                "blocks_shots_2d.py", ["--model", "sigsbee2a", "--f0", "8", "--tmax", "4"],
                json="sigsbee2a.json", figures=("sigsbee2a.png",), gpu=True, runtime="10 min"),
        )),
    "marmousi": dict(
        description="Marmousi 2-D",
        studies=dict(
            convergence_gpu=Study(
                "blocks_convergence_2d.py", ["--model", "marmousi", "--tmax", "3"],
                json="marmousi.json", figures=("marmousi.png",), gpu=True, runtime="10 min"),
            shots_gpu=Study(
                "blocks_shots_2d.py", ["--model", "marmousi", "--f0", "8", "--tmax", "4"],
                json="marmousi.json", figures=("marmousi.png",), gpu=True, runtime="5 min"),
        )),
    "bilayer-inclusion": dict(
        description="bicouche + inclusion carrée (validation LTS)",
        studies=dict(
            shots_gpu=Study(
                "blocks_shots_2d.py", ["--model", "bilayer-inclusion", "--f0", "8", "--tmax", "4"],
                json="bilayer-inclusion.json", figures=("bilayer-inclusion.png",), gpu=True, runtime="5 min",
                note="le README publie ce cas à l'ordre 2 : ajouter --order 2 pour le comparer"),
        )),
    "layers3-slow-inclusion": dict(
        description="3 couches + inclusion lente",
        studies=dict(
            shots_gpu=Study(
                "blocks_shots_2d.py", ["--model", "layers3-slow-inclusion", "--f0", "8", "--tmax", "4"],
                json="layers3-slow-inclusion.json", figures=("layers3-slow-inclusion.png",), gpu=True,
                runtime="5 min"),
        )),
    "disque": dict(
        description="disque rapide 4000 m/s dans 2000 m/s (synthétique)",
        studies=dict(
            hybrid=Study(
                "hybrid_disk_2d.py", [],
                json="disque_4000_dans_2000.json", figures=("disque_4000_dans_2000.png",), runtime="3 min",
                expected=dict(errors={"hybride": (0.0114, 0.0166), "blocs (escalier)": (0.0456, 0.0783)},
                              elements={"hybride": 2116, "blocs (escalier)": 1900})),
        )),
    "onde-plane": dict(
        description="erreur due aux interfaces non conformes (onde plane, contrastes 0,25 à 4)",
        studies=dict(
            interfaces=Study(
                "interface_error_study.py", [],
                json="interface_error_study.json", figures=("interface_error_study.png",), runtime="10 min"),
        )),
}
