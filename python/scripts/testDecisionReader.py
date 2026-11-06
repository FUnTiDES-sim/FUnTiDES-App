import importlib.util, sys

# Load without triggering __init__.py
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

dr = load('decision_reader', 'makutu/acquisition/decision_reader.py')

# --- SEAM: 1 shot, 1 receiver ---
d_src, d_rcv, n = dr.read_decision_file(
    '../../FUnTiDES-data/data/1.Acquisition/SEAM_Foothills.decision.mapped',
    swords=[22, 23, 14], rwords=[24, 25, 13], scalco=1, scalel=1)
print(f'SEAM: n={n}  src={d_src}  rcv={d_rcv}')

# --- test acquisition: 2 shots, 300 receivers each ---
d_src, d_rcv, n = dr.read_decision_file(
    'apps/modeling/examples/inputfiles/acquisition/decision.mapped',
    swords=[22, 23, 15], rwords=[24, 25, 13], scalco=1, scalel=1)
print(f'TestAcq: n={n}')
for k in sorted(d_src): print(f'  {k}: src={d_src[k]}  n_rcv={len(d_rcv[\"rcv_\"+k[3:]])}')

# --- test/SEAM (files we created) ---
d_src, d_rcv, n = dr.read_decision_file(
    '../../FUnTiDES-data/data/test/SEAM_Foothills.decision.mapped',
    swords=[22, 23, 14], rwords=[24, 25, 13], scalco=1, scalel=1)
print(f'test/SEAM: n={n}  src={d_src}  rcv={d_rcv}')
