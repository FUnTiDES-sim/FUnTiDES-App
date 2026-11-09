# seismesh — maillages SEM adaptés à un modèle de vitesses (makutu.mesh.seismesh)

Maillages tout-quadrangles (2-D, coupe (x, z)) ou tout-hexaèdres (3-D), conformes, dont la taille
d'élément suit le modèle de vitesses et la fréquence de la source ; export pour les lecteurs gmsh
de FUnTiDES et modélisation avec makutu (`GmshMesh`, `AcousticSolver`). Issu de `seismesh_agent`
(sans la partie agent LLM).

## Utilisation : modèle SEP + fréquence de la source
```bash
python apps/mesh/seismesh_from_sep.py --vp-file vp.H --f0 8 --order 4 --ppw 5 --out maillage   # [--rho-file rho.H] [--lts]
```
```python
from makutu.mesh import GmshMesh
from makutu.mesh.seismesh import build_mesh_from_sep
r = build_mesh_from_sep("vp.H", f0=8.0, out_dir="maillage", order=4, ppw=5)
mesh = GmshMesh(r["msh"], 4, r["materials"])            # puis AcousticSolver(..., **mesh.solver_kwargs())
```
- `f_max = 2,5 f0` pour une ondelette de Ricker de fréquence de pic `f0` (comme
  `makutu.mesh.suggest_mesh`) ; taille requise `h = N·v/(f_max·ppw)`.
- SEP 2-D (`n3 = 1`) : axe 1 = x, axe 2 = profondeur → quadrangles (`tensor`, `doubling`) ;
  SEP 3-D : axes x, y, z → hexaèdres (`tensor`, `doubling`, `hybrid`, `extrude`).
- Les candidats sont générés et évalués ; le moins coûteux de ceux qui sont valides
  (`ppw` atteint, jacobien > 0,2) est écrit dans `out_dir` :
  - `mesh.msh` : MSH 2.2, **z vers le haut** (hauteur = profondeur max − profondeur), surface
    libre en haut (groupe 1 `free_surface`), autres faces absorbantes (groupe 2 `absorbing`),
    un groupe physique par classe de vitesse (≥ 100) ;
  - `materials.csv` : `physical_tag,vp,rho`, vp = moyenne des échantillons SEP de l'élément ;
  - `report.json` : qualité, échantillonnage, pas de temps CFL, coût du maillage retenu.
- Les profondeurs de makutu (z vers le bas depuis la surface) sont celles du modèle SEP
  (origine `o3` comprise).
- Validation (tests de makutu) : tirs sur maillages à doublements contre le maillage cartésien
  de makutu du même modèle, écart relatif des traces ≤ 3,3 % en 2-D et ≤ 2,5 % en 3-D à
  10 points par longueur d'onde ; il décroît quand le maillage est raffiné.
- Interfaces : des faces d'éléments passent par les interfaces horizontales nettes (saut de
  vitesse > 10 % entre deux échantillons sur au moins la moitié du plan), à mi-chemin des deux
  échantillons. Les interfaces inclinées ou courbes (inclusions) restent traversées : un élément
  à cheval reçoit la moyenne des lenteurs de ses échantillons (`average="slowness"`, défaut,
  conserve les temps de trajet ; `"velocity"` : moyenne des vitesses).
- Modèle à 3 couches et inclusion lente (2-D, 8 pts/λ) : sans faces sur les interfaces, écart
  des traces de 52 à 80 % et arrivées en avance jusqu'à 18 ms ; avec, 6 à 40 % et 8 ms, dont
  5 ms viennent de la position de l'interface (milieu des échantillons, 495 m, contre 500 m
  pour le maillage cartésien de référence) ; le reste, de l'inclusion (marches de 20 m contre
  éléments de 70 m).

Convention des grilles : tableau (ix, iy, iz), z = profondeur vers le bas, z = zmin = surface
libre ; en 2-D, ny = 1. Autres entrées de `VelocityGrid.load` : `.npy`, `.npz`, binaire brut
(+ `.json` de métadonnées).

## Chaîne
1. **Critère** `h_req = N·v_min/(f_max·ppw)` (N = ordre SEM ; N=1 et ppw≈10 pour un critère type DF).
2. **Segmentation optimale** par programmation dynamique de chaque axe en ≤K segments à pas
   constant (min du nombre d'éléments, gradation bornée entre segments).
3. **gmsh** :
   - `tensor` — extrusions structurées point→ligne→surface→volume avec couches : 100 % hexa, jacobien 1.
   - `doubling` — grille tensorielle + **couches de doublement conformes 2:1** (briques type SPECFEM) :
     dès que h_req le permet sur toute la profondeur restante, le pas horizontal double via deux
     sous-couches (gabarit 2D « 4 fines → 2 grossières », 6 quads convexes, extrudé en y puis en x).
     100 % hexa, sans nœud pendant ; maillage construit en numpy puis injecté dans gmsh (entités
     discrètes + groupes physiques). Paramètres : `max_levels`, `trans_layers`.
   - `hybrid` — `doubling` + **3-raffinement local conforme** (arbre 27, gabarits de Schneiders
     reconstruits : sommet 4 hexas, arête 11, face 22 ; équilibrage fort ; jacobien min 0.46 sur
     grille cubique). Le plan de doublement suit un fond h_bg (quantile latéral `q_bg` de h_req) ;
     les anomalies lentes localisées sont raffinées 1:3 dans les zones à niveau constant, les
     transitions 2:1 étant écartées des anomalies (replanification). Fusion conforme vérifiée.
   - `extrude` — quads non structurés (Frontal-Delaunay + Blossom) pilotés par un callback de taille
     sur min_z(h_req), extrudés en couches en z. Pertinent si la lenteur est localisée latéralement.
4. **Évaluation** : jacobien normalisé, rapport d'aspect, ppw effectif par élément (v_min sur les
   nœuds DF de l'élément), pas de temps CFL (écart GLL min), `cost_index ∝ points GLL × nb de pas`,
   efficacité vs borne idéale ∫dV/h_req³, éléments limitant le dt.
5. **Export** : `.msh` (groupes physiques `medium`, `free_surface`, `absorbing`), `.vtk` (vp, ppw,
   jacobien, dt par élément pour ParaView), `_sem.npz` (nœuds, connectivité, vp min/max/moy par élément).

## Recherche automatique
`Session.autopilot` (3-D) et `plane.autopilot_2d` (2-D) balaient les paramètres des stratégies
et classent les candidats par `cost_index` (pas global) ou `cost_index_lts` (pas de temps local
idéal par puissances de 2, sans le surcoût du support de correction mesuré dans FUnTiDES).

## Convention des modèles SEP
Comme makutu, l'échantillon i représente la cellule [o + i d, o + (i+1) d) : le domaine maillé est
[o, o + n d] et les interfaces tombent sur les bords des cellules (`grid_from_sep(cell_centred=True)`,
défaut). La vitesse d'un élément est la moyenne des lenteurs des cellules, pondérée par la part
recouverte ; la taille requise d'un élément est celle de la cellule la plus lente qu'il recouvre.

## 2-D : du grossier au fin (quadtree_2d.py, stratégie `quadtree`)
Grille de base dimensionnée pour la vitesse la plus rapide (blocs de 2 x 2 éléments), puis
découpe 2:1 des blocs tant que leurs éléments dépassent la taille requise : les zones rapides
(sel) gardent de grands éléments, les zones lentes (eau) en ont de petits. Transitions conformes au
niveau du bloc selon ses côtés voisins de blocs plus fins : gabarit de doublement (un côté, deux
côtés opposés), quart raffiné + deux gabarits de coin à 3 quadrangles (deux côtés adjacents, trois
côtés) ; quatre côtés : découpe. La recherche balaie la taille de base (1 à 0,5 fois la plus grande
taille requise, pour que les niveaux base / 2^l tombent près des tailles du modèle) et un décalage
de la grille de base. Validation : modèle SEG/EAGE 2-D sel (f0 = 5 Hz, ordre 4) contre le maillage
cartésien de makutu (20 m) : 15 % d'écart global à 8 pts/λ (3 663 éléments), 5,5 % à 16 pts/λ ;
modèle synthétique à sel (tests) : 9,6 % puis 4,1 %.

## Maillages 2-D (plane.py)
- `tensor` : grille à pas constant par morceaux en x et en z ;
- `doubling` : le pas horizontal double en profondeur par une couche de transition conforme
  (gabarit « 4 fines → 2 grossières », 6 quadrangles) dès que `h_req` le permet sur la suite ;
  `allow_refine` re-raffine sous une couche lente (gabarit retourné, jacobien min ≈ 0,5).

## Résultats (modèle synthétique 5×4×6 km, fmax 10 Hz, ordre 4, 5 ppw)
| stratégie | éléments | dt CFL | cost_index | jacobien min |
|---|---|---|---|---|
| tensor | 49 491 | 1.2 ms | 10.6 | 1.00 |
| doubling (L=3, doublements à 1025 et 3350 m) | 44 160 | 2.7 ms | **4.26** | 0.68 |

Le gain vient surtout du pas de temps : plus de petits éléments dans les zones rapides profondes.

## Raffinement local (hybrid) : quand l'utiliser
Grand domaine 20×16×6 km, 12 poches lentes (2000 m/s) dans la 3e couche, fmax 10 Hz :

| stratégie | éléments | cost (dt global) | cost (LTS idéal) |
|---|---|---|---|
| doubling + segmentation latérale | 869 760 | **160** | 121 |
| hybrid (doubling + 3-raffinement) | 691 272 | 271 | **92** |

−20 % d'éléments, mais les éléments raffinés/de transition se trouvent dans l'encaissant rapide et
divisent le dt global par ~2. → `hybrid` n'est rentable qu'avec un pas de temps local (LTS) ;
sinon `doubling` reste le meilleur choix. Avec `--lts`, la recherche classe selon le coût LTS
(niveaux en puissances de 2, plafonnés par `lts_max_level`) et rapporte `lts_levels`
(nombre d'éléments par niveau, utile pour l'équilibrage de charge).

## Limites / suite
- Doublement global sur tout le plan (pas de doublement local en x,y) ; le nombre de cellules fines
  est arrondi au multiple de 2^(L+1).
- Re-raffinement en profondeur (`allow_refine`) par briques retournées, mais toujours sur tout le plan :
  une inclusion lente localisée impose le niveau fin sur toute la tranche de profondeur concernée.
- `trans_layers=2` peut sous-échantillonner en z (côté gauche de la brique sur toute la hauteur).
- Le pas de temps local de FUnTiDES est à deux niveaux et en 2-D seulement : `cost_index_lts`
  (plusieurs niveaux, 3-D, sans surcoût) est une borne optimiste.
- Modélisation seulement : la FWI de makutu suppose encore une grille cartésienne.
- 2-D : pas de stratégie `hybrid` ni `extrude`.
