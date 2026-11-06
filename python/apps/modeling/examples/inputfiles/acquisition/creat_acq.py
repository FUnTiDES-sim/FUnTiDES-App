# Command that created decision.* from shot_1.sgy and shot_2.sgy (run from FUnTiDES-App/python).
# Decision words: source 22, 23, 14 and receiver 24, 25, 13 (raw integers, no scalar words);
# SEG-Y words read: source 19, 20, 13 (sx, sy, sdepth), receiver 21, 22, 11 (gx, gy, gelev = depth).
#
#   python3 makutu/acquisition/decision_create.py \
#       --from-segy apps/modeling/examples/inputfiles/acquisition/shot_1.sgy \
#                   apps/modeling/examples/inputfiles/acquisition/shot_2.sgy \
#       --out apps/modeling/examples/inputfiles/acquisition/decision --index-words 22,23,14
