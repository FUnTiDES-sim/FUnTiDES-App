# seismesh — maillages SEM adaptés à un modèle de vitesses (makutu.mesh.seismesh)

Maillages tout-quadrangles (2-D, coupe (x, z)) ou tout-hexaèdres (3-D), conformes, dont la taille
d'élément suit le modèle de vitesses et la fréquence de la source ; export pour les lecteurs gmsh
de FUnTiDES et modélisation avec makutu (`GmshMesh`, `AcousticSolver`). Issu de `seismesh_agent`
(sans la partie agent LLM).

> **Suite en 2-D (11-12 novembre 2026).** Un maillage conforme de quadrangles ne peut pas à la fois suivre
> la vitesse localement et rester régulier (niveaux dyadiques : 2,16 fois l'idéal). L'option retenue en
> 2-D est le pavage en blocs réguliers non conformes recollés par SIPG, avec une variante hybride à
> triangles DG le long des interfaces : bibliothèque `makutu.mortar`, applications et résultats dans
> `apps/mortar/`. seismesh reste la voie conforme (2-D et 3-D) pour les lecteurs gmsh de FUnTiDES.

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
- SEP 2-D (`n3 = 1`) : axe 1 = x, axe 2 = profondeur → quadrangles ; stratégies essayées par défaut
  `tensor`, `doubling`, `quadtree`, `bands` (`hybrid` et `adaptive` sur demande, par `strategies`) ;
  SEP 3-D : axes x, y, z → hexaèdres ; par défaut `tensor`, `doubling`, `hybrid`, `refine3tz`,
  `extrude` (`refine3`, `tetsplit` sur demande). Option `--strategies` de `seismesh_from_sep.py`.
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

## 3-D : du grossier au fin (stratégie `refine3tz`)
Grille de base dimensionnée pour la vitesse la plus rapide (`size_factor` × plus grande taille
requise), 3-raffinement (arbre 27) des cellules dont la taille dépasse la taille requise (1/3, 1/9),
puis transitions conformes par les gabarits « à économie d'éléments » de **Hua Tong et Yongjie
Jessica Zhang** :

> H. Tong, Y. J. Zhang, *Element-saving hexahedral 3-refinement templates*, Computer Aided Geometric
> Design (2026) 102550, doi:[10.1016/j.cagd.2026.102550](https://doi.org/10.1016/j.cagd.2026.102550).
> Code d'origine : [CMU-CBML/Element-Saving-Hexahedral-3-Refinement-Templates](https://github.com/CMU-CBML/Element-Saving-Hexahedral-3-Refinement-Templates)
> (licence MIT, Copyright (c) 2026 CMU-CBML).

`templates3_es.py` en est une traduction en Python (méthode 3, par sommets : un sommet est « noir »
quand une cellule plus fine le touche ; 256 codes, 15 classes). Contrairement aux gabarits de
Schneiders (`refine3`, où une cellule bordée par deux ou trois faces raffinées doit être raffinée,
d'où une cascade), tous les codes sont admissibles : un **îlot grossier** (sel) garde ses cellules
de base, les transitions occupant ses cellules de bord. Conditions : équilibrage par contact (deux
feuilles qui se touchent diffèrent d'au plus un niveau, `Refine3Mesh.balance_touch`) ; le corps
rapide doit faire au moins 3 cellules de base d'épaisseur pour qu'une cellule reste intacte.
Qualité : jacobien normalisé min ≈ 0,23 dans les transitions, 1 ailleurs.

Modèle `salt3d` (2 × 2 × 1,5 km, eau 200 m à 1500 m/s, sédiments 2000 + 0,3 (z − 200) m/s, ellipsoïde
de sel 1400 × 1400 × 700 m à 4500 m/s ; f0 = 5 Hz, ordre 3, 8 pts/λ) : 73 540 hexaèdres (tailles 44 m
hors du sel, 133 m dans le sel), 19 s de maillage, contre 657 096 hexaèdres irréguliers pour
`tetsplit`.

## 2-D : comparaison avec gmsh (10 novembre 2026)
Modèles de référence de `FUnTiDES-data/SEP` (seg2d, Marmousi, Sigsbee 2A), f0 = 5 Hz, ordre 4, 8 pts/λ ;
coût = éléments / pas de temps (`cost_index`) ; meilleur maillage valide de chaque approche :

| modèle (éléments idéaux) | seismesh retenu | gmsh, quadrangles frontaux + recombinaison | gmsh, triangles subdivisés |
|---|---|---|---|
| seg2d (1 704) | `quadtree` 3 663, dt 1,3 ms, coût 0,18 | 5 326, dt 0,45 ms, coût 0,77 | 6 120, coût 0,93 |
| Marmousi (3 513) | `tensor` 7 348, dt 1,1 ms, coût 0,43 | 11 590, dt 0,55 ms, coût 1,36 | 9 904, coût 1,11 |
| Sigsbee 2A (31 106) | `doubling` 49 440, dt 1,1 ms, coût 2,88 | 89 468, dt 0,20 ms, coût 28,1 | 104 884, coût 17,4 |

gmsh suit la taille demandée à ±40 % près : pour échantillonner partout, il faut lui demander 0,6 fois
la taille requise (taille médiane obtenue 0,6 fois la taille requise, contre 0,73 à 0,91 pour
seismesh), et ses éléments irréguliers ont des arêtes courtes qui divisent le pas de temps par 2 à 5.
L'algorithme quasi structuré de gmsh (11) ne suit pas la carte de taille. → En 2-D on garde les
stratégies structurées ; la recherche automatique choisit `quadtree` quand un corps rapide occupe
peu de place (seg2d), `tensor` / `doubling` quand la vitesse croît surtout avec la profondeur.

### Essais sans suite
- Tolérance au bord des corps rapides (laisser un bloc grossier déborder dans les sédiments) : le
  contraste sel / sédiments (≈ 2,2) l'interdit sans descendre sous 4 pts/λ.
- Maillage gmsh par régions suivant les interfaces principales (seuils de vitesse, contours
  simplifiés) : mêmes défauts que gmsh ci-dessus.
- Quadtree dont les nœuds sont collés sur les interfaces (Zhang et Bajaj 2006) puis optimisés
  (Knupp) : sur seg2d 4 544 éléments, dt 0,70 ms, écart au tir 16,0 %, contre 3 663 éléments,
  1,3 ms et 15,1 % pour le quadtree simple. Il reste un anneau d'éléments fins dans le sel (blocs 2 x 2).
- Maillage structuré + gmsh dans les corps rapides (trou autour du sel, nœuds du bord imposés, raccord
  conforme) : gmsh grossit bien le sel, mais ses éléments irréguliers divisent le pas de temps par 2
  à 4 (seg2d : coût 0,28 à 0,33 contre 0,18 ; Sigsbee : 10,9 contre 2,9) ; même avec le pas local,
  pas mieux que le quadtree seul (0,19 contre 0,17).
- Déplacement des nœuds du quadtree vers la taille requise (r-adaptation, matrice cible de Knupp) :
  à nombre d'éléments fixé, il ne fait qu'égaliser la finesse (gain de 5 % sur le pas de temps) et
  n'arrive pas à rendre valide un quadtree plus grossier (connectivité figée).
- Régions maillées en grilles régulières raccordées par des bandes gmsh (`patches`, d'abord en
  quadrangles recombinés, puis en triangles subdivisés comme la validation du pas local de FUnTiDES) :
  sur le bicouche de cette validation (SEP à 10 m), 1 847 éléments dont certains 1,36 fois trop
  grands, contre 1 716 pour le quadtree, valide ; les raccords sont larges et gmsh n'y tient pas la taille.
- Zones de taille requise maillées en grilles régulières, raccordées par des transitions gmsh
  optimisées (`zones2d.py`, `mesh_zones_2d`) : échoue sur le sel mince et les sédiments à gradient ;
  module gardé, hors de la recherche automatique et de la ligne de commande.
L'extraction des interfaces (`interfaces.py`) et l'optimisation des nœuds en 2-D (`smoothing.py`)
sont gardées. La stratégie `bands` (bandes de profondeur à pas horizontal propre, raccordées par des
rangées de gabarits 3 -> 1, `bands2d.py`) reste dans la recherche par défaut ; pour un rapport de
taille inférieur à 2 entre bandes, les cellules du raccord se cisaillent (jusqu'à 1,6 fois trop
grandes) et sont alors rejetées par l'évaluation. Ce qui a été retenu : la recherche balaie aussi des tailles de base du quadtree au-delà
de la plus grande taille requise (1,2 à 1,6) et ``ppw_min`` (option ``--ppw-min``) accepte un
échantillonnage un peu inférieur à la cible dans quelques éléments. La stratégie 2-D `hybrid`
(doublement + 3-raffinement local) n'est plus essayée par défaut : jamais retenue (coût 2 à 5 fois celui
de la meilleure) et lente (140 s sur Marmousi contre 6 s pour les autres) ; elle reste disponible par
``strategies`` / ``--strategies``.

## Maillages 2-D (plane.py et modules 2-D)
- `tensor` : grille à pas constant par morceaux en x et en z ;
- `doubling` : le pas horizontal double en profondeur par une couche de transition conforme
  (gabarit « 4 fines → 2 grossières », 6 quadrangles) dès que `h_req` le permet sur la suite ;
  `allow_refine` re-raffine sous une couche lente (gabarit retourné, jacobien min ≈ 0,5) ;
- `quadtree` (`quadtree_2d.py`) : du grossier au fin, voir plus haut ;
- `bands` (`bands2d.py`) : bandes de profondeur régulières à pas horizontal propre ;
- `hybrid` : `doubling` + 3-raffinement local (`refine3_2d.py`, `Refine3Mesh2D`), sur demande ;
- `adaptive` : quadrangles non structurés gmsh (frontal-Delaunay + recombinaison) de taille
  `size_factor × h_req` (`mesh_adaptive_2d`), sur demande ; c'est l'approche gmsh comparée plus haut ;
- valeurs du modèle sur la géométrie réelle des éléments, y compris déformés (`elements2d.py`).

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
- 2-D : pas de stratégie `extrude` ; `hybrid` n'est essayée que sur demande (jamais retenue).
- 2-D : pour suivre la vitesse localement sans éléments déformés, voir `makutu.mortar` (blocs non
  conformes), en tête de ce document.
