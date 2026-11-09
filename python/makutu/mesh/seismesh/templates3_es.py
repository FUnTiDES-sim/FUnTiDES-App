"""Gabarits de 3-raffinement hexaédrique « à économie d'éléments » (méthode 3, par sommets).

Référence — ce module est une traduction en Python de leur travail ; merci de le citer :

    Hua Tong, Yongjie Jessica Zhang, « Element-saving hexahedral 3-refinement templates »,
    Computer Aided Geometric Design, 2026, article 102550. doi:10.1016/j.cagd.2026.102550

    @article{tong2026element,
      title={Element-saving hexahedral 3-refinement templates},
      author={Tong, Hua and Zhang, Yongjie Jessica},
      journal={Computer Aided Geometric Design}, pages={102550}, year={2026}, publisher={Elsevier}}

Code d'origine : https://github.com/CMU-CBML/Element-Saving-Hexahedral-3-Refinement-Templates
(commit 7c68119, fichier src/removeHangingNodeMethod3.cpp), Copyright (c) 2026 CMU-CBML, licence MIT
reproduite ci-dessous. La table ``ROTATIONS`` (code de sommets → classe, permutation des sommets), les
primitives et les classes 1 à 14 sont traduites mécaniquement du C++ (lecture du texte, rien n'a été
compilé ni exécuté) ; la classe 15 (27 sous-cubes) est écrite directement. Seule la construction des
hexaèdres est reprise : l'arbre, l'équilibrage et l'export de leur code ne le sont pas.

Principe : un sommet d'une cellule est « noir » quand une cellule plus fine (1/3) le touche. Le code
de la cellule est un octet, bit 7 = sommet 0, …, bit 0 = sommet 7 (ordre gmsh : face inférieure
0-3, face supérieure 4-7). ``ROTATIONS[code] = (classe, perm)`` ; les sommets réordonnés
``E[i] = coins[perm[i]]`` sont passés au gabarit de la classe. Les nœuds intérieurs ne sont pas sur
un réseau rationnel simple : la fusion des nœuds se fait sur coordonnées arrondies.

    MIT License

    Copyright (c) 2026 CMU-CBML

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
# flake8: noqa  (generated tables and literal translation of the original templates)
from __future__ import annotations

import numpy as np

ROTATIONS = {
    0: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    1: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    2: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    3: (2, (7, 3, 2, 6, 4, 0, 1, 5)),
    4: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    5: (1, (4, 7, 6, 5, 0, 3, 2, 1)),
    6: (2, (5, 4, 7, 6, 1, 0, 3, 2)),
    7: (4, (5, 4, 7, 6, 1, 0, 3, 2)),
    8: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    9: (2, (4, 0, 3, 7, 5, 1, 2, 6)),
    10: (1, (7, 6, 5, 4, 3, 2, 1, 0)),
    11: (4, (6, 5, 4, 7, 2, 1, 0, 3)),
    12: (2, (4, 7, 6, 5, 0, 3, 2, 1)),
    13: (4, (7, 6, 5, 4, 3, 2, 1, 0)),
    14: (4, (4, 7, 6, 5, 0, 3, 2, 1)),
    15: (3, (7, 6, 5, 4, 3, 2, 1, 0)),
    16: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    17: (2, (3, 2, 6, 7, 0, 1, 5, 4)),
    18: (1, (7, 3, 2, 6, 4, 0, 1, 5)),
    19: (4, (3, 2, 6, 7, 0, 1, 5, 4)),
    20: (1, (3, 2, 6, 7, 0, 1, 5, 4)),
    21: (2, (3, 2, 6, 7, 0, 1, 5, 4)),
    22: (2, (5, 4, 7, 6, 1, 0, 3, 2)),
    23: (7, (3, 2, 6, 7, 0, 1, 5, 4)),
    24: (1, (0, 3, 7, 4, 1, 2, 6, 5)),
    25: (4, (4, 0, 3, 7, 5, 1, 2, 6)),
    26: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    27: (6, (3, 2, 6, 7, 0, 1, 5, 4)),
    28: (2, (5, 1, 0, 4, 6, 2, 3, 7)),
    29: (8, (4, 0, 3, 7, 5, 1, 2, 6)),
    30: (4, (4, 7, 6, 5, 0, 3, 2, 1)),
    31: (10, (3, 2, 6, 7, 0, 1, 5, 4)),
    32: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    33: (1, (3, 2, 6, 7, 0, 1, 5, 4)),
    34: (2, (6, 7, 3, 2, 5, 4, 0, 1)),
    35: (4, (7, 3, 2, 6, 4, 0, 1, 5)),
    36: (1, (6, 2, 1, 5, 7, 3, 0, 4)),
    37: (1, (4, 0, 3, 7, 5, 1, 2, 6)),
    38: (4, (2, 1, 5, 6, 3, 0, 4, 7)),
    39: (6, (7, 3, 2, 6, 4, 0, 1, 5)),
    40: (1, (4, 0, 3, 7, 5, 1, 2, 6)),
    41: (2, (4, 0, 3, 7, 5, 1, 2, 6)),
    42: (2, (2, 1, 5, 6, 3, 0, 4, 7)),
    43: (8, (7, 3, 2, 6, 4, 0, 1, 5)),
    44: (2, (4, 7, 6, 5, 0, 3, 2, 1)),
    45: (4, (7, 6, 5, 4, 3, 2, 1, 0)),
    46: (7, (4, 7, 6, 5, 0, 3, 2, 1)),
    47: (10, (2, 1, 5, 6, 3, 0, 4, 7)),
    48: (2, (2, 6, 7, 3, 1, 5, 4, 0)),
    49: (4, (2, 6, 7, 3, 1, 5, 4, 0)),
    50: (4, (6, 7, 3, 2, 5, 4, 0, 1)),
    51: (3, (3, 2, 6, 7, 0, 1, 5, 4)),
    52: (2, (3, 0, 1, 2, 7, 4, 5, 6)),
    53: (4, (2, 6, 7, 3, 1, 5, 4, 0)),
    54: (8, (6, 7, 3, 2, 5, 4, 0, 1)),
    55: (10, (5, 4, 7, 6, 1, 0, 3, 2)),
    56: (2, (2, 6, 7, 3, 1, 5, 4, 0)),
    57: (7, (4, 0, 3, 7, 5, 1, 2, 6)),
    58: (4, (6, 7, 3, 2, 5, 4, 0, 1)),
    59: (10, (4, 0, 3, 7, 5, 1, 2, 6)),
    60: (9, (4, 7, 6, 5, 0, 3, 2, 1)),
    61: (11, (4, 0, 3, 7, 5, 1, 2, 6)),
    62: (11, (2, 1, 5, 6, 3, 0, 4, 7)),
    63: (5, (4, 0, 3, 7, 5, 1, 2, 6)),
    64: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    65: (1, (7, 6, 5, 4, 3, 2, 1, 0)),
    66: (1, (5, 6, 2, 1, 4, 7, 3, 0)),
    67: (2, (7, 3, 2, 6, 4, 0, 1, 5)),
    68: (2, (1, 0, 4, 5, 2, 3, 7, 6)),
    69: (2, (1, 0, 4, 5, 2, 3, 7, 6)),
    70: (4, (6, 2, 1, 5, 7, 3, 0, 4)),
    71: (8, (5, 4, 7, 6, 1, 0, 3, 2)),
    72: (1, (5, 1, 0, 4, 6, 2, 3, 7)),
    73: (2, (7, 6, 5, 4, 3, 2, 1, 0)),
    74: (1, (7, 3, 2, 6, 4, 0, 1, 5)),
    75: (4, (6, 5, 4, 7, 2, 1, 0, 3)),
    76: (4, (1, 0, 4, 5, 2, 3, 7, 6)),
    77: (7, (7, 6, 5, 4, 3, 2, 1, 0)),
    78: (6, (6, 2, 1, 5, 7, 3, 0, 4)),
    79: (10, (1, 0, 4, 5, 2, 3, 7, 6)),
    80: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    81: (2, (7, 4, 0, 3, 6, 5, 1, 2)),
    82: (1, (5, 4, 7, 6, 1, 0, 3, 2)),
    83: (4, (3, 2, 6, 7, 0, 1, 5, 4)),
    84: (2, (5, 6, 2, 1, 4, 7, 3, 0)),
    85: (9, (3, 2, 6, 7, 0, 1, 5, 4)),
    86: (4, (6, 2, 1, 5, 7, 3, 0, 4)),
    87: (11, (5, 4, 7, 6, 1, 0, 3, 2)),
    88: (1, (7, 6, 5, 4, 3, 2, 1, 0)),
    89: (4, (4, 0, 3, 7, 5, 1, 2, 6)),
    90: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    91: (6, (3, 2, 6, 7, 0, 1, 5, 4)),
    92: (4, (1, 0, 4, 5, 2, 3, 7, 6)),
    93: (11, (7, 6, 5, 4, 3, 2, 1, 0)),
    94: (6, (6, 2, 1, 5, 7, 3, 0, 4)),
    95: (12, (3, 2, 6, 7, 0, 1, 5, 4)),
    96: (2, (1, 5, 6, 2, 0, 4, 7, 3)),
    97: (2, (1, 5, 6, 2, 0, 4, 7, 3)),
    98: (4, (1, 5, 6, 2, 0, 4, 7, 3)),
    99: (7, (7, 3, 2, 6, 4, 0, 1, 5)),
    100: (4, (5, 6, 2, 1, 4, 7, 3, 0)),
    101: (4, (5, 6, 2, 1, 4, 7, 3, 0)),
    102: (3, (6, 2, 1, 5, 7, 3, 0, 4)),
    103: (10, (7, 3, 2, 6, 4, 0, 1, 5)),
    104: (2, (2, 3, 0, 1, 6, 7, 4, 5)),
    105: (9, (4, 0, 3, 7, 5, 1, 2, 6)),
    106: (4, (1, 5, 6, 2, 0, 4, 7, 3)),
    107: (11, (7, 3, 2, 6, 4, 0, 1, 5)),
    108: (8, (1, 0, 4, 5, 2, 3, 7, 6)),
    109: (11, (1, 0, 4, 5, 2, 3, 7, 6)),
    110: (10, (4, 7, 6, 5, 0, 3, 2, 1)),
    111: (5, (7, 3, 2, 6, 4, 0, 1, 5)),
    112: (4, (3, 0, 1, 2, 7, 4, 5, 6)),
    113: (8, (2, 6, 7, 3, 1, 5, 4, 0)),
    114: (6, (6, 7, 3, 2, 5, 4, 0, 1)),
    115: (10, (1, 5, 6, 2, 0, 4, 7, 3)),
    116: (7, (5, 6, 2, 1, 4, 7, 3, 0)),
    117: (11, (3, 0, 1, 2, 7, 4, 5, 6)),
    118: (10, (3, 0, 1, 2, 7, 4, 5, 6)),
    119: (5, (5, 4, 7, 6, 1, 0, 3, 2)),
    120: (4, (3, 0, 1, 2, 7, 4, 5, 6)),
    121: (11, (2, 6, 7, 3, 1, 5, 4, 0)),
    122: (6, (6, 7, 3, 2, 5, 4, 0, 1)),
    123: (12, (4, 0, 3, 7, 5, 1, 2, 6)),
    124: (11, (5, 6, 2, 1, 4, 7, 3, 0)),
    125: (13, (4, 0, 3, 7, 5, 1, 2, 6)),
    126: (12, (4, 7, 6, 5, 0, 3, 2, 1)),
    127: (14, (4, 0, 3, 7, 5, 1, 2, 6)),
    128: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    129: (1, (4, 0, 3, 7, 5, 1, 2, 6)),
    130: (1, (0, 1, 2, 3, 4, 5, 6, 7)),
    131: (2, (6, 5, 4, 7, 2, 1, 0, 3)),
    132: (1, (4, 5, 1, 0, 7, 6, 2, 3)),
    133: (1, (3, 2, 6, 7, 0, 1, 5, 4)),
    134: (2, (6, 2, 1, 5, 7, 3, 0, 4)),
    135: (4, (5, 4, 7, 6, 1, 0, 3, 2)),
    136: (2, (0, 3, 7, 4, 1, 2, 6, 5)),
    137: (4, (0, 3, 7, 4, 1, 2, 6, 5)),
    138: (2, (0, 3, 7, 4, 1, 2, 6, 5)),
    139: (7, (0, 3, 7, 4, 1, 2, 6, 5)),
    140: (4, (5, 1, 0, 4, 6, 2, 3, 7)),
    141: (6, (7, 6, 5, 4, 3, 2, 1, 0)),
    142: (8, (4, 7, 6, 5, 0, 3, 2, 1)),
    143: (10, (0, 3, 7, 4, 1, 2, 6, 5)),
    144: (2, (0, 1, 2, 3, 4, 5, 6, 7)),
    145: (4, (7, 4, 0, 3, 6, 5, 1, 2)),
    146: (2, (0, 1, 2, 3, 4, 5, 6, 7)),
    147: (8, (3, 2, 6, 7, 0, 1, 5, 4)),
    148: (2, (3, 7, 4, 0, 2, 6, 5, 1)),
    149: (4, (7, 4, 0, 3, 6, 5, 1, 2)),
    150: (9, (0, 1, 2, 3, 4, 5, 6, 7)),
    151: (11, (3, 2, 6, 7, 0, 1, 5, 4)),
    152: (4, (3, 7, 4, 0, 2, 6, 5, 1)),
    153: (3, (4, 0, 3, 7, 5, 1, 2, 6)),
    154: (4, (3, 7, 4, 0, 2, 6, 5, 1)),
    155: (10, (6, 5, 4, 7, 2, 1, 0, 3)),
    156: (7, (3, 7, 4, 0, 2, 6, 5, 1)),
    157: (10, (5, 1, 0, 4, 6, 2, 3, 7)),
    158: (11, (5, 1, 0, 4, 6, 2, 3, 7)),
    159: (5, (3, 2, 6, 7, 0, 1, 5, 4)),
    160: (1, (1, 2, 3, 0, 5, 6, 7, 4)),
    161: (1, (4, 5, 1, 0, 7, 6, 2, 3)),
    162: (2, (6, 7, 3, 2, 5, 4, 0, 1)),
    163: (4, (7, 3, 2, 6, 4, 0, 1, 5)),
    164: (1, (4, 7, 6, 5, 0, 3, 2, 1)),
    165: (1, (3, 2, 6, 7, 0, 1, 5, 4)),
    166: (4, (2, 1, 5, 6, 3, 0, 4, 7)),
    167: (6, (7, 3, 2, 6, 4, 0, 1, 5)),
    168: (2, (4, 5, 1, 0, 7, 6, 2, 3)),
    169: (4, (0, 3, 7, 4, 1, 2, 6, 5)),
    170: (9, (0, 3, 7, 4, 1, 2, 6, 5)),
    171: (11, (6, 5, 4, 7, 2, 1, 0, 3)),
    172: (4, (5, 1, 0, 4, 6, 2, 3, 7)),
    173: (6, (7, 6, 5, 4, 3, 2, 1, 0)),
    174: (11, (4, 7, 6, 5, 0, 3, 2, 1)),
    175: (12, (0, 3, 7, 4, 1, 2, 6, 5)),
    176: (4, (0, 1, 2, 3, 4, 5, 6, 7)),
    177: (6, (0, 1, 2, 3, 4, 5, 6, 7)),
    178: (7, (0, 1, 2, 3, 4, 5, 6, 7)),
    179: (10, (0, 1, 2, 3, 4, 5, 6, 7)),
    180: (4, (0, 1, 2, 3, 4, 5, 6, 7)),
    181: (6, (0, 1, 2, 3, 4, 5, 6, 7)),
    182: (11, (6, 7, 3, 2, 5, 4, 0, 1)),
    183: (12, (0, 1, 2, 3, 4, 5, 6, 7)),
    184: (8, (0, 1, 2, 3, 4, 5, 6, 7)),
    185: (10, (2, 6, 7, 3, 1, 5, 4, 0)),
    186: (11, (0, 1, 2, 3, 4, 5, 6, 7)),
    187: (5, (0, 1, 2, 3, 4, 5, 6, 7)),
    188: (11, (3, 7, 4, 0, 2, 6, 5, 1)),
    189: (12, (5, 1, 0, 4, 6, 2, 3, 7)),
    190: (13, (0, 1, 2, 3, 4, 5, 6, 7)),
    191: (14, (0, 1, 2, 3, 4, 5, 6, 7)),
    192: (2, (0, 4, 5, 1, 3, 7, 6, 2)),
    193: (2, (1, 2, 3, 0, 5, 6, 7, 4)),
    194: (2, (0, 4, 5, 1, 3, 7, 6, 2)),
    195: (9, (7, 3, 2, 6, 4, 0, 1, 5)),
    196: (4, (0, 4, 5, 1, 3, 7, 6, 2)),
    197: (4, (0, 4, 5, 1, 3, 7, 6, 2)),
    198: (7, (6, 2, 1, 5, 7, 3, 0, 4)),
    199: (11, (6, 2, 1, 5, 7, 3, 0, 4)),
    200: (4, (4, 5, 1, 0, 7, 6, 2, 3)),
    201: (8, (0, 3, 7, 4, 1, 2, 6, 5)),
    202: (4, (4, 5, 1, 0, 7, 6, 2, 3)),
    203: (11, (0, 3, 7, 4, 1, 2, 6, 5)),
    204: (3, (4, 5, 1, 0, 7, 6, 2, 3)),
    205: (10, (7, 6, 5, 4, 3, 2, 1, 0)),
    206: (10, (6, 2, 1, 5, 7, 3, 0, 4)),
    207: (5, (0, 3, 7, 4, 1, 2, 6, 5)),
    208: (4, (1, 2, 3, 0, 5, 6, 7, 4)),
    209: (7, (7, 4, 0, 3, 6, 5, 1, 2)),
    210: (4, (1, 2, 3, 0, 5, 6, 7, 4)),
    211: (11, (7, 4, 0, 3, 6, 5, 1, 2)),
    212: (8, (0, 4, 5, 1, 3, 7, 6, 2)),
    213: (11, (1, 2, 3, 0, 5, 6, 7, 4)),
    214: (11, (0, 4, 5, 1, 3, 7, 6, 2)),
    215: (13, (3, 2, 6, 7, 0, 1, 5, 4)),
    216: (6, (4, 5, 1, 0, 7, 6, 2, 3)),
    217: (10, (1, 2, 3, 0, 5, 6, 7, 4)),
    218: (6, (4, 5, 1, 0, 7, 6, 2, 3)),
    219: (12, (1, 2, 3, 0, 5, 6, 7, 4)),
    220: (10, (3, 7, 4, 0, 2, 6, 5, 1)),
    221: (5, (7, 6, 5, 4, 3, 2, 1, 0)),
    222: (12, (6, 2, 1, 5, 7, 3, 0, 4)),
    223: (14, (3, 2, 6, 7, 0, 1, 5, 4)),
    224: (4, (2, 3, 0, 1, 6, 7, 4, 5)),
    225: (4, (2, 3, 0, 1, 6, 7, 4, 5)),
    226: (8, (1, 5, 6, 2, 0, 4, 7, 3)),
    227: (11, (1, 5, 6, 2, 0, 4, 7, 3)),
    228: (6, (5, 6, 2, 1, 4, 7, 3, 0)),
    229: (6, (5, 6, 2, 1, 4, 7, 3, 0)),
    230: (10, (0, 4, 5, 1, 3, 7, 6, 2)),
    231: (12, (7, 3, 2, 6, 4, 0, 1, 5)),
    232: (7, (4, 5, 1, 0, 7, 6, 2, 3)),
    233: (11, (4, 5, 1, 0, 7, 6, 2, 3)),
    234: (11, (2, 3, 0, 1, 6, 7, 4, 5)),
    235: (13, (7, 3, 2, 6, 4, 0, 1, 5)),
    236: (10, (2, 3, 0, 1, 6, 7, 4, 5)),
    237: (12, (7, 6, 5, 4, 3, 2, 1, 0)),
    238: (5, (4, 7, 6, 5, 0, 3, 2, 1)),
    239: (14, (7, 3, 2, 6, 4, 0, 1, 5)),
    240: (3, (0, 1, 2, 3, 4, 5, 6, 7)),
    241: (10, (7, 4, 0, 3, 6, 5, 1, 2)),
    242: (10, (6, 7, 3, 2, 5, 4, 0, 1)),
    243: (5, (1, 5, 6, 2, 0, 4, 7, 3)),
    244: (10, (5, 6, 2, 1, 4, 7, 3, 0)),
    245: (12, (5, 6, 2, 1, 4, 7, 3, 0)),
    246: (5, (6, 7, 3, 2, 5, 4, 0, 1)),
    247: (14, (5, 4, 7, 6, 1, 0, 3, 2)),
    248: (10, (4, 5, 1, 0, 7, 6, 2, 3)),
    249: (5, (4, 5, 1, 0, 7, 6, 2, 3)),
    250: (12, (4, 5, 1, 0, 7, 6, 2, 3)),
    251: (14, (4, 5, 1, 0, 7, 6, 2, 3)),
    252: (5, (3, 7, 4, 0, 2, 6, 5, 1)),
    253: (14, (7, 6, 5, 4, 3, 2, 1, 0)),
    254: (14, (4, 7, 6, 5, 0, 3, 2, 1)),
    255: (15, (0, 1, 2, 3, 4, 5, 6, 7)),
}

TW, FW = [8, 5, 2], [4, 5, 1]

def interpolatePoint(p1, w1, p2, w2):
    t = w1 + w2
    return tuple(p1[i] * w1 / t + p2[i] * w2 / t for i in range(3))
def interpolate3Points(p1, w1, p2, w2, p3, w3):
    t = w1 + w2 + w3
    return tuple(p1[i] * w1 / t + p2[i] * w2 / t + p3[i] * w3 / t for i in range(3))
def interpolate4Points(p1, w1, p2, w2, p3, w3, p4, w4):
    t = w1 + w2 + w3 + w4
    return tuple(p1[i] * w1 / t + p2[i] * w2 / t + p3[i] * w3 / t + p4[i] * w4 / t for i in range(3))

def template_hexes(cls, E):
    """Hexahedra (lists of 8 points, gmsh order) of class ``cls`` for the 8 points E (rotated corners)."""
    hexes = []
    def emit(h):
        hexes.append(h)
    def noBlackDot(positions):
        emit(list(positions))
    def twoBlackDots(eightPos):
        pts = [None] * 64
        pts[0:8] = list(eightPos)
        pts[8] = interpolatePoint(eightPos[0], 2, eightPos[3], 1)
        pts[9] = interpolate3Points(eightPos[1], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[10] = interpolate3Points(eightPos[2], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        pts[11] = interpolatePoint(eightPos[3], 2, eightPos[0], 1)
        pts[12] = interpolate3Points(eightPos[4], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[13] = interpolate3Points(eightPos[5], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[14] = interpolate3Points(eightPos[6], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        pts[15] = interpolate3Points(eightPos[7], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        emit([pts[0], pts[1], pts[9], pts[8], pts[4], pts[5], pts[13], pts[12]])
        emit([pts[8], pts[9], pts[10], pts[11], pts[12], pts[13], pts[14], pts[15]])
        emit([pts[9], pts[1], pts[2], pts[10], pts[13], pts[5], pts[6], pts[14]])
        emit([pts[11], pts[10], pts[2], pts[3], pts[15], pts[14], pts[6], pts[7]])
        emit([pts[12], pts[13], pts[14], pts[15], pts[4], pts[5], pts[6], pts[7]])
    def twoBlackDots2(eightPos):
        pts = [None] * 64
        TW2 = [7, 5, 3]
        FW2 = [24, 40, -15, 71]
        pts[0:8] = list(eightPos)
        pts[8] = interpolatePoint(eightPos[0], 2, eightPos[3], 1)
        pts[9] = interpolate3Points(eightPos[1], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[10] = interpolate3Points(eightPos[2], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        pts[11] = interpolatePoint(eightPos[3], 2, eightPos[0], 1)
        pts[12] = interpolate3Points(eightPos[4], TW2[0], eightPos[3], TW2[1], eightPos[0], TW2[2])
        pts[13] = interpolate4Points(eightPos[0], FW2[0], eightPos[3], FW2[1], eightPos[4], FW2[2], eightPos[5], FW2[3])
        pts[14] = interpolate4Points(eightPos[3], FW2[0], eightPos[0], FW2[1], eightPos[7], FW2[2], eightPos[6], FW2[3])
        pts[15] = interpolate3Points(eightPos[7], TW2[0], eightPos[0], TW2[1], eightPos[3], TW2[2])
        emit([pts[0], pts[1], pts[9], pts[8], pts[4], pts[5], pts[13], pts[12]])
        emit([pts[8], pts[9], pts[10], pts[11], pts[12], pts[13], pts[14], pts[15]])
        emit([pts[9], pts[1], pts[2], pts[10], pts[13], pts[5], pts[6], pts[14]])
        emit([pts[11], pts[10], pts[2], pts[3], pts[15], pts[14], pts[6], pts[7]])
        emit([pts[12], pts[13], pts[14], pts[15], pts[4], pts[5], pts[6], pts[7]])
    def fourBlackDots(eightPos):
        pts = [None] * 64
        pts[0:8] = list(eightPos)
        pts[8] = interpolatePoint(eightPos[0], 2, eightPos[1], 1)
        pts[9] = interpolatePoint(eightPos[1], 2, eightPos[0], 1)
        pts[10] = interpolatePoint(eightPos[0], 2, eightPos[3], 1)
        pts[11] = interpolatePoint(eightPos[0], 2, eightPos[2], 1)
        pts[12] = interpolatePoint(eightPos[1], 2, eightPos[3], 1)
        pts[13] = interpolatePoint(eightPos[1], 2, eightPos[2], 1)
        pts[14] = interpolatePoint(eightPos[3], 2, eightPos[0], 1)
        pts[15] = interpolatePoint(eightPos[3], 2, eightPos[1], 1)
        pts[16] = interpolatePoint(eightPos[2], 2, eightPos[0], 1)
        pts[17] = interpolatePoint(eightPos[2], 2, eightPos[1], 1)
        pts[18] = interpolatePoint(eightPos[3], 2, eightPos[2], 1)
        pts[19] = interpolatePoint(eightPos[2], 2, eightPos[3], 1)
        pts[20] = interpolate4Points(eightPos[0], FW[0], eightPos[1], FW[1], eightPos[3], FW[1], eightPos[4], FW[2])
        pts[21] = interpolate4Points(eightPos[1], FW[0], eightPos[0], FW[1], eightPos[2], FW[1], eightPos[5], FW[2])
        pts[22] = interpolate4Points(eightPos[3], FW[0], eightPos[0], FW[1], eightPos[2], FW[1], eightPos[7], FW[2])
        pts[23] = interpolate4Points(eightPos[2], FW[0], eightPos[1], FW[1], eightPos[3], FW[1], eightPos[6], FW[2])
        pts[24] = interpolate3Points(eightPos[4], TW[0], eightPos[1], TW[1], eightPos[0], TW[2])
        pts[25] = interpolate3Points(eightPos[5], TW[0], eightPos[0], TW[1], eightPos[1], TW[2])
        pts[26] = interpolate3Points(eightPos[5], TW[0], eightPos[2], TW[1], eightPos[1], TW[2])
        pts[27] = interpolate3Points(eightPos[6], TW[0], eightPos[1], TW[1], eightPos[2], TW[2])
        pts[28] = interpolate3Points(eightPos[6], TW[0], eightPos[3], TW[1], eightPos[2], TW[2])
        pts[29] = interpolate3Points(eightPos[7], TW[0], eightPos[2], TW[1], eightPos[3], TW[2])
        pts[30] = interpolate3Points(eightPos[7], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        pts[31] = interpolate3Points(eightPos[4], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        emit([pts[0], pts[8], pts[11], pts[10], pts[4], pts[24], pts[20], pts[31]])
        emit([pts[8], pts[9], pts[12], pts[11], pts[24], pts[25], pts[21], pts[20]])
        emit([pts[9], pts[1], pts[13], pts[12], pts[25], pts[5], pts[26], pts[21]])
        emit([pts[10], pts[11], pts[15], pts[14], pts[31], pts[20], pts[22], pts[30]])
        emit([pts[11], pts[12], pts[16], pts[15], pts[20], pts[21], pts[23], pts[22]])
        emit([pts[12], pts[13], pts[17], pts[16], pts[21], pts[26], pts[27], pts[23]])
        emit([pts[14], pts[15], pts[18], pts[3], pts[30], pts[22], pts[29], pts[7]])
        emit([pts[15], pts[16], pts[19], pts[18], pts[22], pts[23], pts[28], pts[29]])
        emit([pts[16], pts[17], pts[2], pts[19], pts[23], pts[27], pts[6], pts[28]])
        emit([pts[20], pts[21], pts[23], pts[22], pts[24], pts[25], pts[28], pts[29]])
        emit([pts[31], pts[20], pts[22], pts[30], pts[4], pts[24], pts[29], pts[7]])
        emit([pts[21], pts[26], pts[27], pts[23], pts[25], pts[5], pts[6], pts[28]])
        emit([pts[24], pts[25], pts[28], pts[29], pts[4], pts[5], pts[6], pts[7]])
    def fourBlackDots2(eightPos):
        pts = [None] * 64
        pts[0:8] = list(eightPos)
        pts[8] = interpolatePoint(eightPos[0], 2, eightPos[3], 1)
        pts[9] = interpolate3Points(eightPos[1], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[10] = interpolate3Points(eightPos[2], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        pts[11] = interpolatePoint(eightPos[3], 2, eightPos[0], 1)
        pts[12] = interpolate3Points(eightPos[4], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[13] = interpolate3Points(eightPos[5], TW[0], eightPos[3], TW[1], eightPos[0], TW[2])
        pts[14] = interpolate3Points(eightPos[6], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        pts[15] = interpolate3Points(eightPos[7], TW[0], eightPos[0], TW[1], eightPos[3], TW[2])
        emit([pts[0], pts[1], pts[9], pts[8], pts[4], pts[5], pts[13], pts[12]])
        emit([pts[8], pts[9], pts[10], pts[11], pts[12], pts[13], pts[14], pts[15]])
        emit([pts[11], pts[10], pts[2], pts[3], pts[15], pts[14], pts[6], pts[7]])
        twoBlackDots2([pts[6], pts[2], pts[1], pts[5], pts[14], pts[10], pts[9], pts[13]])
        twoBlackDots2([pts[5], pts[4], pts[7], pts[6], pts[13], pts[12], pts[15], pts[14]])
    def threeBlackDots(eightPos):
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolate3Points(eightPos[4], 3, eightPos[2], 4, eightPos[0], 9), interpolate3Points(eightPos[5], 3, eightPos[3], 4, eightPos[1], 9), interpolate3Points(eightPos[6], 3, eightPos[0], 4, eightPos[2], 9), interpolate3Points(eightPos[7], 3, eightPos[1], 4, eightPos[3], 9)]
        twoBlackDots([allP[0], allP[8], allP[11], allP[3], allP[4], allP[12], allP[15], allP[7]])
        twoBlackDots([allP[3], allP[11], allP[10], allP[2], allP[7], allP[15], allP[14], allP[6]])
        noBlackDot([allP[0], allP[1], allP[9], allP[8], allP[4], allP[5], allP[13], allP[12]])
        noBlackDot([allP[1], allP[2], allP[10], allP[9], allP[5], allP[6], allP[14], allP[13]])
        noBlackDot([allP[12], allP[13], allP[14], allP[15], allP[4], allP[5], allP[6], allP[7]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    eightPos = E
    if cls == 1:
        noBlackDot(E)
    elif cls == 2:
        twoBlackDots(E)
    elif cls == 3:
        fourBlackDots(E)
    elif cls == 4:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        twoBlackDots([allP[0], allP[8], allP[11], allP[3], allP[4], allP[12], allP[15], allP[7]])
        twoBlackDots([allP[3], allP[11], allP[10], allP[2], allP[7], allP[15], allP[14], allP[6]])
        noBlackDot([allP[0], allP[1], allP[9], allP[8], allP[4], allP[5], allP[13], allP[12]])
        noBlackDot([allP[1], allP[2], allP[10], allP[9], allP[5], allP[6], allP[14], allP[13]])
        noBlackDot([allP[12], allP[13], allP[14], allP[15], allP[4], allP[5], allP[6], allP[7]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 5:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[6], 1), interpolatePoint(eightPos[5], 3, eightPos[7], 1), interpolatePoint(eightPos[6], 3, eightPos[4], 1), interpolatePoint(eightPos[7], 3, eightPos[5], 1)]
        fourBlackDots([allP[0], allP[3], allP[7], allP[4], allP[8], allP[11], allP[15], allP[12]])
        fourBlackDots([allP[3], allP[2], allP[6], allP[7], allP[11], allP[10], allP[14], allP[15]])
        twoBlackDots([allP[4], allP[5], allP[1], allP[0], allP[12], allP[13], allP[9], allP[8]])
        twoBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 6:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        threeBlackDots([allP[7], allP[4], allP[0], allP[3], allP[15], allP[12], allP[8], allP[11]])
        threeBlackDots([allP[2], allP[6], allP[7], allP[3], allP[10], allP[14], allP[15], allP[11]])
        noBlackDot([allP[0], allP[1], allP[9], allP[8], allP[4], allP[5], allP[13], allP[12]])
        noBlackDot([allP[1], allP[2], allP[10], allP[9], allP[5], allP[6], allP[14], allP[13]])
        noBlackDot([allP[12], allP[13], allP[14], allP[15], allP[4], allP[5], allP[6], allP[7]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 7:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        twoBlackDots([allP[0], allP[8], allP[11], allP[3], allP[4], allP[12], allP[15], allP[7]])
        threeBlackDots([allP[6], allP[7], allP[3], allP[2], allP[14], allP[15], allP[11], allP[10]])
        noBlackDot([allP[0], allP[1], allP[9], allP[8], allP[4], allP[5], allP[13], allP[12]])
        twoBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        noBlackDot([allP[12], allP[13], allP[14], allP[15], allP[4], allP[5], allP[6], allP[7]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 8:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        threeBlackDots([allP[3], allP[7], allP[4], allP[0], allP[11], allP[15], allP[12], allP[8]])
        twoBlackDots([allP[3], allP[11], allP[10], allP[2], allP[7], allP[15], allP[14], allP[6]])
        twoBlackDots([allP[4], allP[5], allP[1], allP[0], allP[12], allP[13], allP[9], allP[8]])
        noBlackDot([allP[1], allP[2], allP[10], allP[9], allP[5], allP[6], allP[14], allP[13]])
        noBlackDot([allP[12], allP[13], allP[14], allP[15], allP[4], allP[5], allP[6], allP[7]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 9:
        fourBlackDots2(E)
    elif cls == 10:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        threeBlackDots([allP[7], allP[4], allP[0], allP[3], allP[15], allP[12], allP[8], allP[11]])
        fourBlackDots([allP[3], allP[2], allP[6], allP[7], allP[11], allP[10], allP[14], allP[15]])
        twoBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        noBlackDot([allP[4], allP[5], allP[1], allP[0], allP[12], allP[13], allP[9], allP[8]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
        twoBlackDots([allP[6], allP[5], allP[4], allP[7], allP[14], allP[13], allP[12], allP[15]])
    elif cls == 11:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        threeBlackDots([allP[3], allP[7], allP[4], allP[0], allP[11], allP[15], allP[12], allP[8]])
        threeBlackDots([allP[6], allP[7], allP[3], allP[2], allP[14], allP[15], allP[11], allP[10]])
        twoBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        twoBlackDots([allP[4], allP[5], allP[1], allP[0], allP[12], allP[13], allP[9], allP[8]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
        noBlackDot([allP[6], allP[5], allP[4], allP[7], allP[14], allP[13], allP[12], allP[15]])
    elif cls == 12:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[6], 1), interpolatePoint(eightPos[5], 3, eightPos[7], 1), interpolatePoint(eightPos[6], 3, eightPos[4], 1), interpolatePoint(eightPos[7], 3, eightPos[5], 1)]
        threeBlackDots([allP[7], allP[4], allP[0], allP[3], allP[15], allP[12], allP[8], allP[11]])
        fourBlackDots([allP[3], allP[2], allP[6], allP[7], allP[11], allP[10], allP[14], allP[15]])
        threeBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        noBlackDot([allP[4], allP[5], allP[1], allP[0], allP[12], allP[13], allP[9], allP[8]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 13:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[6], 1), interpolatePoint(eightPos[5], 3, eightPos[7], 1), interpolatePoint(eightPos[6], 3, eightPos[4], 1), interpolatePoint(eightPos[7], 3, eightPos[5], 1)]
        threeBlackDots([allP[3], allP[7], allP[4], allP[0], allP[11], allP[15], allP[12], allP[8]])
        threeBlackDots([allP[6], allP[7], allP[3], allP[2], allP[14], allP[15], allP[11], allP[10]])
        threeBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        threeBlackDots([allP[5], allP[1], allP[0], allP[4], allP[13], allP[9], allP[8], allP[12]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
    elif cls == 14:
        allP = [eightPos[0], eightPos[1], eightPos[2], eightPos[3], eightPos[4], eightPos[5], eightPos[6], eightPos[7], interpolatePoint(eightPos[0], 3, eightPos[2], 1), interpolatePoint(eightPos[1], 3, eightPos[3], 1), interpolatePoint(eightPos[2], 3, eightPos[0], 1), interpolatePoint(eightPos[3], 3, eightPos[1], 1), interpolatePoint(eightPos[4], 3, eightPos[2], 1), interpolatePoint(eightPos[5], 3, eightPos[3], 1), interpolatePoint(eightPos[6], 3, eightPos[0], 1), interpolatePoint(eightPos[7], 3, eightPos[1], 1)]
        fourBlackDots([allP[3], allP[7], allP[4], allP[0], allP[11], allP[15], allP[12], allP[8]])
        fourBlackDots([allP[6], allP[7], allP[3], allP[2], allP[14], allP[15], allP[11], allP[10]])
        threeBlackDots([allP[2], allP[1], allP[5], allP[6], allP[10], allP[9], allP[13], allP[14]])
        threeBlackDots([allP[5], allP[1], allP[0], allP[4], allP[13], allP[9], allP[8], allP[12]])
        noBlackDot([allP[8], allP[9], allP[10], allP[11], allP[12], allP[13], allP[14], allP[15]])
        fourBlackDots([allP[7], allP[6], allP[5], allP[4], allP[15], allP[14], allP[13], allP[12]])
    elif cls == 15:                        # all vertices marked: 27 sub-cubes (trilinear thirds)
        def P(a, b, c):
            u, v, w = a / 3, b / 3, c / 3
            wts = [(1-u)*(1-v)*(1-w), u*(1-v)*(1-w), u*v*(1-w), (1-u)*v*(1-w), (1-u)*(1-v)*w, u*(1-v)*w, u*v*w, (1-u)*v*w]
            return tuple(sum(wt * E[i][d] for i, wt in enumerate(wts)) for d in range(3))
        for c in range(3):
            for b in range(3):
                for a in range(3):
                    emit([P(a, b, c), P(a+1, b, c), P(a+1, b+1, c), P(a, b+1, c), P(a, b, c+1), P(a+1, b, c+1), P(a+1, b+1, c+1), P(a, b+1, c+1)])
    else:
        raise ValueError(cls)
    return hexes


_CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])


def cell_code(black) -> int:
    """Code de la cellule à partir des 8 drapeaux « sommet noir » (ordre gmsh)."""
    return sum(128 >> i for i, b in enumerate(black) if b)


def cell_hexes(lo, size, code):
    """Hexaèdres (liste de 8 points (x, y, z)) remplaçant la cellule [lo, lo + size] de code donné."""
    lo, size = np.asarray(lo, float), np.asarray(size, float)
    cls, perm = ROTATIONS[code]
    corners = [tuple(lo + size * c) for c in _CORNERS]
    return template_hexes(cls, [corners[p] for p in perm])


def two_level_mesh(origin, h, refined, decimals=9):
    """Maillage conforme d'une grille de cellules de taille ``h`` dont les cellules ``refined``
    (booléens (nx, ny, nz)) sont 3-raffinées : les autres reçoivent le gabarit de leur code.

    Retourne (nœuds (n, 3), hexaèdres (ne, 8) ordre gmsh, classe de la cellule d'origine (ne,))."""
    refined = np.asarray(refined, bool)
    nx, ny, nz = refined.shape
    h = np.broadcast_to(np.asarray(h, float), (3,))
    pad = np.pad(refined, 1)
    black = np.zeros((nx + 1, ny + 1, nz + 1), bool)          # vertex touched by a refined cell
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                black |= pad[a:a + nx + 1, b:b + ny + 1, c:c + nz + 1]
    nodes, index, hexes, cls = [], {}, [], []
    scale = float(np.max(h))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                code = 255 if refined[i, j, k] else cell_code(
                    [black[i + a, j + b, k + c] for a, b, c in _CORNERS])
                lo = np.asarray(origin, float) + h * (i, j, k)
                for hx in cell_hexes(lo, h, code):
                    ids = []
                    for p in hx:
                        key = tuple(np.round(np.asarray(p) / scale, decimals))
                        if key not in index:
                            index[key] = len(nodes)
                            nodes.append(p)
                        ids.append(index[key])
                    hexes.append(ids)
                    cls.append(ROTATIONS[code][0])
    return np.array(nodes, float), np.array(hexes, int), np.array(cls, int)
