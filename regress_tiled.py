#!/usr/bin/env python3
import argparse, os, subprocess, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'model'))
from attn_model import tiled_model, fp_reference_tiled
FP_TOL = 0.06


def build(n, maxt):
    out = os.path.join(ROOT, 'build', f'tiled{n}_{maxt}'); os.makedirs(out, exist_ok=True)
    rtl = os.path.join(ROOT, 'rtl')
    srcs = [os.path.join(rtl, f) for f in ('pe_db.sv', 'attn_core_db.sv', 'osm.sv', 'normalizer.sv', 'attn_tiled.sv')]
    cmd = ['verilator', '--binary', '--timing', '--assert', '-Wno-fatal', '-Wno-WIDTH', '-Wno-GENUNNAMED',
           '-I' + rtl, f'-GN={n}', f'-GMAXT={maxt}', '--top-module', 'tb_tiled', '-Mdir', out, '-o', 'Vtb',
           *srcs, os.path.join(ROOT, 'tb', 'tb_tiled.sv')]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:]); raise SystemExit('build failed')
    return os.path.join(out, 'Vtb'), out


def gen(rng, n, maxt, kind):
    def mat(rows, d):
        if d == 0: return rng.integers(-128, 128, (rows, n))
        if d == 1: return np.clip(np.round(rng.normal(0, 30, (rows, n))), -128, 127).astype(int)
        if d == 2: return rng.integers(-128, 128, (rows, n)) * (rng.random((rows, n)) < 0.4)
        return rng.integers(-8, 9, (rows, n))
    nt = int(rng.integers(1, maxt + 1)); d = int(rng.integers(0, 4))
    cfg = dict(ntiles=nt, causal=int(rng.random() < 0.5), qblk=int(rng.integers(0, maxt)),
               sm_mult=int(rng.integers(1, 65536)), sm_shift=int(rng.integers(10, 30)), th_qk=0, th_pv=0,
               tile_lo=int(rng.integers(0, maxt)) if rng.random() < 0.3 else 0)
    if kind == 'gate':
        cfg['th_qk'], cfg['th_pv'] = int(rng.integers(0, 20)), int(rng.integers(0, 20))
    Q, K, V = mat(n, d), mat(nt * n, d), mat(nt * n, d)
    if kind == 'zero': Q[:] = 0
    if kind == 'max': Q[:] = 127; K[:] = -128; V[:] = 127
    if kind == 'sat': cfg['sm_mult'], cfg['sm_shift'] = 65535, 0
    if kind == 'hot': cfg['sm_mult'], cfg['sm_shift'] = int(rng.integers(30000, 65536)), int(rng.integers(0, 6))
    if kind == 'diag0': cfg.update(causal=1, qblk=0)
    if kind == 'window': cfg.update(causal=1, qblk=int(rng.integers(1, maxt)), ntiles=maxt, tile_lo=int(rng.integers(0, maxt))); K, V = mat(maxt * n, d), mat(maxt * n, d)
    if kind == 'full': cfg.update(ntiles=maxt, causal=0, tile_lo=0); K, V = mat(maxt * n, d), mat(maxt * n, d)
    return Q, K, V, cfg


def run(n, maxt, tests, seed, quiet=False):
    rng = np.random.default_rng(seed)
    sim, work = build(n, maxt)
    kinds = ['zero', 'max', 'sat', 'diag0', 'full'] * 4
    vec = [gen(rng, n, maxt, k) for k in kinds]
    while len(vec) < tests:
        vec.append(gen(rng, n, maxt, rng.choice(['rand', 'rand', 'rand', 'gate', 'full', 'hot', 'window'])))
    flags = [(int(i > 0 and rng.random() < 0.08), int(rng.random() < 0.1)) for i in range(len(vec))]
    with open(os.path.join(work, 'vectors.txt'), 'w') as f:
        f.write(f'{len(vec)}\n')
        for (Q, K, V, c), (rst, poke) in zip(vec, flags):
            f.write(f"{c['ntiles']} {c['causal']} {c['qblk']} {c['tile_lo']} {c['sm_mult']} {c['sm_shift']} "
                    f"{c['th_qk']} {c['th_pv']} {rst} {poke}\n")
            for M in (Q, K, V):
                f.write(' '.join(str(int(x)) for x in M.flatten()) + '\n')
    log = subprocess.run([sim, '+VEC=vectors.txt', '+RES=results.txt'], cwd=work, capture_output=True, text=True)
    if 'ASSERT' in log.stdout + log.stderr or log.returncode:
        print(log.stdout[-3000:], log.stderr[-3000:]); raise SystemExit('assertion/sim failure')
    L = open(os.path.join(work, 'results.txt')).read().split('\n')
    fails, lat_by_tiles, li, fp_err = [], {}, 0, []
    for idx, (Q, K, V, c) in enumerate(vec):
        h = list(map(int, L[li].split()[1:])); li += 1
        rows = [list(map(int, L[li + r].split())) for r in range(n)]; li += n
        lr = [row[0] for row in rows]; O = np.array([row[1:] for row in rows])
        m = tiled_model(Q, K, V, c, n, maxt)
        e = []
        if not np.array_equal(O, m['O']): e.append('O')
        if lr != m['l']: e.append('l')
        if h[3] != m['mac_qk'] or h[4] != m['mac_pv']: e.append('perf')
        if h[6] != m['te']: e.append('tiles')
        if c['th_qk'] == 0 and c['th_pv'] == 0:
            fe = float(np.abs(O / 256.0 - fp_reference_tiled(Q, K, V, c, n)).max() / 128)
            fp_err.append(fe)
            if fe > FP_TOL: e.append(f'fp({fe:.3f})')
        if e: fails.append((idx, c, e))
        if c['th_qk'] == 0 and c['th_pv'] == 0:
            lat_by_tiles.setdefault(m['te'], set()).add(h[1])
    res = dict(N=n, MAXT=maxt, tests=len(vec), fails=len(fails),
               fp_checked=len(fp_err), fp_err_max=round(max(fp_err), 4) if fp_err else None,
               latency_by_tiles={k: sorted(v) for k, v in sorted(lat_by_tiles.items())})
    if not quiet:
        print(json.dumps(res))
        for f in fails[:8]: print('  FAIL', f)
    return res, fails


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--maxt', type=int, default=8)
    ap.add_argument('--tests', type=int, default=500); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    r, f = run(a.n, a.maxt, a.tests, a.seed)
    sys.exit(1 if f else 0)
