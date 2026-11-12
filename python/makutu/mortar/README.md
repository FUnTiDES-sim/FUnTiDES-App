# makutu.mortar : blocs réguliers non conformes (prototype 2-D)

Option B retenue le 2026-11-11 : au lieu d'un maillage conforme de quadrangles (transitions déformées),
le domaine est pavé de **blocs rectangulaires réguliers**, chacun à sa propre taille de maille,
recollés par une **pénalité intérieure symétrique (SIPG)** sur les segments d'interface. SEM continu
dans chaque bloc, rapport de tailles quelconque entre blocs, aucun élément déformé.

NumPy / SciPy seulement (importable sans GPU). Prototype de validation avant un portage dans
FUnTiDES (sem2d : ajout d'un noyau de faces SIPG sur les segments d'interface, cf. plus bas).

Document de référence (maths, implémentation, validation, prototype hybride) :
[doc/mortar-maths-et-implementation.md](doc/mortar-maths-et-implementation.md).

## Modules

| module | contenu |
|---|---|
| `solver2d.py` | `Mortar2D` : SEM 2-D acoustique (GLL, masse diagonale), SIPG sur les interfaces, saute-mouton, bords paraxiaux + éponge, propriétés par élément ou aux nœuds (`nodal`), `dt_max()` exact (2/sqrt(lambda_max(M^-1 K))) |
| `blocks2d.py` | `plan_blocks` : pavage optimal d'un modèle SEP 2-D (programmation dynamique sur l'arbre des coupes binaires au milieu) ; `uniform_plan` : grille unique à la plus petite taille requise |
| `plots2d.py` | figures de contrôle des pavages |

## Validation (tests/unitTests/test_mortar2d.py, figures dans tests/figures/test_mortar2d)

- K symétrique, semi-définie positive avec la pénalité par défaut
  alpha = 1,25 N / (2 (N + 1)) (alpha minimal mesuré 0,22 / 0,32 / 0,37 aux ordres 2 / 4 / 6) ;
  coût en pas de temps d'une interface conforme : 8 % à l'ordre 4 (alpha = 1 coûtait 37 %).
- Énergie discrète conservée à 1e-15 près (6000 pas, 4 blocs de rapports 1,37 à 3,5, contrastes de vitesse
  et de densité, propriétés aux nœuds).
- Réflexion parasite d'une interface en milieu homogène (ordre 4, côté grossier à 5 points / longueur
  d'onde) : 1e-5 (r = 1), 2e-5 (r = 1,37), 3e-4 (r = 2) de l'onde directe.
- Convergence spectrale (h, 1,5 h) ; écart à une référence fine dominé par le côté grossier.

## Comptes d'éléments (ordre 4, 5 points GLL / longueur d'onde, f0 = 8 Hz ; bicouche : ordre 2, fmax 5 Hz, 6 ppw)

| modèle | idéal | blocs | ratio | meilleur conforme précédent |
|---|---|---|---|---|
| seg2d | 1704 | 2345 (50 blocs) | 1,38 | bandes 2808, tensor 4042 |
| Marmousi | 3513 | 4960 (67) | 1,41 | bandes 6294, doubling 7224 |
| Sigsbee 2A | 31106 | 36159 (276) | 1,16 | bandes 44314, doubling 49440 |
| Sigsbee + gaz | 31548 | 36722 (298) | 1,16 | bandes 45474, doubling 51088 |
| bicouche + inclusion | 1209 | 1356 (27) | 1,12 | quadtree 1716 |
| 3 couches + inclusion lente | 328 | 492 (24) | 1,50 | bandes 606, doubling 624 |

Pavage en 0,1 à 9 s (l'autopilote conforme : 7 s à 11 min). Le sel reçoit de grands éléments, la poche
de gaz un bloc fin ajusté ; seuls les blocs à cheval sur le bord du sel gardent la taille des sédiments.

## Tirs contre une référence calée sur les cellules SEP (propriétés aux nœuds GLL)

Le modèle SEP doit être lu comme un **modèle continu** (`grid_sampler`, interpolation bilinéaire des
échantillons placés aux centres des cellules), le même pour la référence et pour tous les maillages.
Lu cellule par cellule, il présente un petit saut à chaque bord de cellule ; des nœuds placés n'importe
où dans les cellules voient alors un modèle localement décalé, et l'erreur ne converge plus
(Marmousi : 1,6 à 7,2 % selon le maillage, sans ordre).

Vérifications : pavage Marmousi en milieu homogène, écart 0,3 % (5 ppw) puis 0,07 % (7 ppw) ; K p = 0 aux
nœuds intérieurs pour p linéaire (1e-14) ; modèle analytique lisse : blocs 0,75 / 0,20 / 0,12 / 0,04 %
de 5 à 8 ppw, uniforme 0,74 % à 5 ppw avec deux fois plus d'éléments.

Écart L2 relatif des traces (récepteurs à 80 m, offsets > 300 m, 2 s), modèle bilinéaire :

| modèle | maillage | éléments | dt stable | écart |
|---|---|---|---|---|
| Marmousi | uniforme 5 ppw | 9686 | 1,33 ms | 2,31 % |
| Marmousi | blocs 5 ppw | 4960 | 1,46 ms | 2,39 % |
| Marmousi | blocs 6 ppw | 6917 | 1,40 ms | 1,42 % |
| seg2d | uniforme 5 ppw | 3698 | 1,39 ms | 2,30 % |
| seg2d | blocs 5 ppw | 2345 | 1,14 ms | 2,78 % |
| seg2d | blocs 7 ppw | 4388 | 0,90 ms | 1,58 % |

À précision égale, les blocs demandent environ 2,2 fois moins de calcul sur Marmousi et 1,3 fois moins
sur seg2d (pas de temps global ; davantage avec un pas de temps par bloc). Le palier vers 1,5-2 % vient
des vraies discontinuités (sel, fond de mer) qui traversent les éléments, pour tous les maillages.
Le pas de temps des blocs peut être plus petit que celui de la grille uniforme : l'arrondi
ceil(W / h) dans les petits blocs donne des éléments plus minces (prochain point : coût par bloc
n_e / dt_bloc dans la programmation dynamique, naturel avec le pas de temps local).

## Coût en calcul (rank, dt_min)

Pas de temps d'un bloc : dt = C min(hx, hz) / vp_max, C = 0,1044 à l'ordre 4 (valeur propre exacte d'une
grille régulière ; l'estimation est exacte pour un bloc seul et surestime d'environ 8 % avec interfaces,
à cause de la pénalité). `rank="lts"` minimise la somme des éléments / dt_bloc (pas de temps propre à
chaque bloc) ; `dt_min` refuse les blocs plus lents qu'un pas global visé.

Travail (millions d'éléments-pas par seconde simulée, ordre 4, 5 ppw, f0 = 8 Hz) :

| modèle | uniforme | blocs, pas global (dt_min = dt uniforme) | blocs, pas par bloc (rank="lts") |
|---|---|---|---|
| seg2d | 2,66 | 1,61 (x1,65) | 0,88 (x3,0) |
| Marmousi | 7,38 | 3,12 (x2,4) | 1,73 (x4,3) |
| Sigsbee 2A | 51,3 | 30,5 (x1,7) | 10,9 (x4,7) |

## FUnTiDES (GPU) : funtides2d.py, apps/mortar/blocks_shots_2d.py

FUnTiDES (branche dev-2d, non installé dans FUnTiDES-install) : `Solver::setInterfaceOperator(dofs, P, G,
gamma, m, n_q)` dans SEMsolver2D, forme SIPG factorisée par point de Gauss (deux noyaux : produits P.u et
G.u, puis lignes avec ajouts atomiques) ; les arêtes d'un seul élément qui contiennent un nœud intérieur
(arêtes d'interface) ne sont plus amorties. `FuntidesBlocks2D` construit le ModelUnstruct 2-D à nœuds
dupliqués avec la numérotation de Mortar2D. FUnTiDES (float32) reproduit le prototype à 1e-6 près
(tests/integratedTests/test_mortar_funtides.py).

Tirs FUnTiDES, pas de temps global (plan_blocks_global), propriétés aux nœuds, référence calée sur les
cellules SEP ; écart L2 des traces (offsets > 300 m) / de l'instantané à 0,6 T :

| modèle | blocs 5 ppw | blocs 6 ppw | uniforme 5 ppw |
|---|---|---|---|
| Sigsbee 2A | 37 596 él., 5,5 / 5,4 % | 53 249 él., 4,5 / 4,3 % | 67 840 él., 6,7 / 5,9 % |
| Marmousi | 5 476 él., 7,0 / 4,8 % | 7 642 él., 1,1 / 3,1 % | 9 686 él., 2,7 / 3,4 % |
| seg2d | 2 724 él., 3,0 / 8,2 % | 3 707 él., 2,6 / 5,4 % | 3 698 él., 2,3 / 4,9 % |
| bicouche + inclusion (ordre 2) | 1 398 él. (6 ppw), 8,8 / 9,3 % | 1 939 él. (7 ppw), 7,3 / 7,6 % | 11 250 él., 10,3 / 11,4 % |
| 3 couches + inclusion lente | 674 él., 4,7 / 12,9 % | 789 él., 4,7 / 11,0 % | 850 él., 4,8 / 12,3 % |

Temps GPU (portable, horloge variable : échauffement puis meilleur de trois) : Sigsbee 0,18 ms par pas
(blocs 5) contre 0,23 ms (uniforme), soit 1,0 s contre 1,2 s pour 6 s simulées à leurs pas stables, avec
une meilleure précision. Sous 10 000 éléments, le GPU est limité par la latence des lancements de noyaux
et les temps ne distinguent pas les maillages. L'opérateur d'interface coûte encore l'équivalent de 1 à 2
éléments par segment : prochains leviers, un noyau par équipe (un lancement, pas de tableau
intermédiaire) et surtout le pas de temps propre à chaque bloc (travail estimé divisé par 3 à 4,7).

## Noyau d'interface et récepteurs (FUnTiDES, 2026-11-11)

- Opérateur d'interface structuré (`Mortar2D.interface_faces`) : traces et dérivées normales aux nœuds de
  face (dérivée le long de la ligne d'élément normale), interpolées aux points de Gauss ; un fil par
  segment. Sigsbee (4663 segments) : 0,024 ms par pas, contre 0,106 ms avec les lignes pleines en deux
  noyaux et 0,187 ms avec une équipe par segment. Blocs 0,128 ms par pas contre 0,258 ms pour la grille
  uniforme (37 596 contre 67 840 éléments).
- `pyfuntides.solver.FieldSampler` : sommes pondérées aux récepteurs sur le GPU ; 392 récepteurs coûtent
  0,019 ms par pas, contre environ 0,9 ms en lisant le champ (mémoire unifiée) depuis Python.

## Pas de temps par bloc (lts2d.py, prototype abandonné pour l'instant)

LF-LTS de Diaz et Grote appliqué récursivement par niveaux (bloc de niveau l : pas dt / 2^l), version
stabilisée (Grote, Michel, Sauter), niveaux sur la stabilité exacte (plus grande valeur propre de
l'opérateur restreint à chaque groupe de nœuds, interfaces comprises : `assign_levels`).
- Stabilité : il faut promouvoir les éléments des deux côtés d'une interface au niveau le plus fin
  (pénalité en 1 / h du côté fin) et graduer les niveaux (au plus un niveau d'écart entre nœuds couplés :
  un saut de deux niveaux diverge, mesuré) ; l'énergie reste alors bornée.
- Gain : au pas de stabilité, le saute-mouton a déjà une erreur temporelle de l'ordre de 5 à 8 % sur 2 à
  3 s (comparé à dt / 4) ; agrandir le grand pas dégrade la précision. Au pas que demande la précision
  (celui de la grille uniforme), le LTS ne gagne rien sur Marmousi et 4 à 14 % sur Sigsbee, avec un cas
  instable : non porté dans FUnTiDES. Le pavage `plan_blocks_global` (pas de temps global) est retenu.
- Figure : tests/figures/mortar/lts_diagnostic.png.

## Dans les solveurs makutu (mesh2d.py)

`BlocksMesh2D(vp_file, f0=...)` : pavage, maillage, modèle FUnTiDES ; `solver_kwargs()` pour
`AcousticSolver` (localisateur des sources et récepteurs, crochet `solver_setup` qui pose l'opérateur
d'interface après chaque `compute_fe_init`, propriétés aux nœuds). Pour la FWI, la variable est la grille
du modèle SEP : `updateVelocityModel` réinterpole les nœuds (interpolation bilinéaire B), le gradient est
calculé aux nœuds puis ramené sur la grille par la transposée de B. Tests
(tests/integratedTests/test_mortar_makutu_solver.py) : tir AcousticSolver = tir direct à 1,4e-5 ; gradient
= différences finies à 0,2 % (paramétrisations c et 1/c^2).

## Interfaces nettes et validation de la modélisation

- `plan_blocks(z_cuts="auto")` coupe d'abord le modèle en tranches sur les interfaces horizontales nettes
  (saut > 10 % sur la moitié de la largeur, `sharp_interfaces`, interfaces distantes d'au moins une taille
  d'élément), puis optimise chaque tranche : aucun élément ne traverse ces interfaces.
- Bords nets locaux (`edge_cuts`, `force_jump=0.3`) : dans chaque rectangle, la programmation dynamique
  essaie aussi des coupes sur les bords nets (saut > 10 % sur la moitié du rectangle) et interdit de
  garder en un seul bloc un rectangle traversé par un saut > 30 % : un corps net aligné sur les axes est
  bordé par des bords de blocs. Diagnostic (bicouche + inclusion carrée à 600 m/s, ordre 2, 6 ppw) : une
  grille uniforme dont les éléments tombent par hasard sur les bords de l'inclusion (40 m) fait 1,6 %,
  une grille non alignée (38,7 m) 7,7 % ; les blocs passent de 8,5 % à 4,1 % (1254 éléments). Une
  gradation de la taille requise (`grading`, taille lipschitzienne) n'aide qu'à 9 ppw (2,5 %) : non
  activée par défaut.
- `Mortar2D(nodal_inset=...)` (BlocksMesh2D : un demi-pas de grille) : les nœuds d'un bord de bloc lisent
  le modèle à l'intérieur de leur bloc ; une interface sur un bord de bloc est vue d'un seul côté de part
  et d'autre (saut exact, possible car les blocs ne partagent aucun nœud).
- tests/integratedTests/test_mortar_modeling_validation.py (makutu AcousticSolver + FUnTiDES, GPU) :
  solution exacte 2-D à travers 9 blocs de rapports 1,15 à 1,9 (ordres 2 à 4, source sur un coin de
  blocs) < 3e-3 ; surface libre, modèle en couches, contre grille fine coupée sur l'interface : 2,6 %
  (18 % avant les coupes sur interfaces, 9,6 % avec les coupes mais une référence qui lissait le saut) ;
  8 s simulées sans croissance.

## Références et lecture du modèle SEP

Une grille SEP peut se lire « en marches » (constante par cellule : référence exacte = grille uniforme d'un
élément par cellule, propriétés par élément) ou « lissée » (interpolation bilinéaire : référence = grille
uniforme fine, propriétés aux nœuds). Les deux lectures diffèrent de 2 % (bicouche) à 8 % (Sigsbee, bords
du sel courbes) sur les traces : la référence doit lire le modèle comme le maillage testé.
Écart des traces (offsets > 300 m), tests/figures/mortar/references_*.png :
- bicouche + inclusion carrée (ordre 2, sauts sur les bords des cellules, que les blocs suivent) : contre
  la référence en marches, blocs 1,9 / 0,85 / 0,34 / 0,12 % de 6 à 12 ppw ; grille uniforme non alignée
  14 % / 7 % (6 / 9 ppw). Le « plafond » de 3,8 % était l'écart entre les deux références.
- Sigsbee : contre la référence lissée à une cellule (3,8 millions d'éléments), blocs 5,1 / 3,9 / 2,7 /
  2,4 % de 5 à 8 ppw, uniforme 5,6 / 2,4 % (5 / 7 ppw) ; l'ancienne référence (3 cellules) en était à
  2,8 %. Le reste (~2,4 %, commun aux blocs et à la grille uniforme) vient des bords nets du sel à
  l'échelle de la cellule, que tout maillage plus grossier que la cellule voit de la même façon.
- Marmousi : les blocs coupent sur le fond de mer et le voient comme un saut net ; la lecture lissée en
  fait une rampe d'une cellule : blocs à 0,7 % de la référence lissée à 2 cellules (8 ppw), 1,9 % de
  celle à une cellule ; la grille uniforme (7 ppw) l'inverse (1,9 / 0,6 %).
Lecture appliquée par BlocksMesh2D : lissée, sauf sur les bords de blocs qui portent un saut net
(lecture d'un seul côté, saut exact).

## Portage FUnTiDES (estimation)

Le DG 3-D de FUnTiDES apparie les faces par nœuds partagés (FaceConnectivityUnstruct : une face = deux
éléments) : pas de faces h-non conformes sans réécriture. Le SEM 2-D (sem2d) est la bonne base : boucle
élément + ajouts atomiques dans un vecteur global. À ajouter : maillage à nœuds dupliqués aux bords de
blocs (injection depuis Python), table des segments d'interface (élément, arête, paramètres des deux
côtés), un noyau Kokkos de flux SIPG par segment (traces et dérivées normales interpolées aux points de
Gauss), le pas de temps par bloc pour le pas de temps local.
