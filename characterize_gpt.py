#!/usr/bin/env python3
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, os.path.dirname(__file__))
from attn_model import tiled_model, HP
from train_tiny_gpt import GPT, load_corpus

N = MAXT = None
LAT = {}


def quant(x):
    s = max(float(np.abs(x).max()), 1e-12) / 127.0
    return np.clip(np.round(x / s), -128, 127).astype(np.int64), s


def scale_cfg(c):
    c2 = c * math.log2(math.e) * 2 ** HP['XF']
    shift = int(min(63, max(0, math.floor(math.log2(65535 / c2)))))
    return int(min(65535, max(1, round(c2 * 2 ** shift)))), shift


class Stats:
    def __init__(s): s.__dict__.update(mac=0, mac_dense=0, tiles=0, tiles_dense=0, cycles=0, heads=0,
                                       q_zero=0, q_tot=0, p_zero=0, p_tot=0, err=[])


def hw_attention(th_qk=0, th_pv=0, st=None, floor=False, layer=0):
    def fn(q, k, v):
        B, H, T, hd = q.shape
        out = torch.zeros_like(q)
        tri = np.triu_indices(T, 1)
        for b in range(B):
            for h in range(H):
                Qf, Kf, Vf = (t[b, h].double().numpy() for t in (q, k, v))
                Q, sq = quant(Qf); K, sk = quant(Kf); V, sv = quant(Vf)
                c = sq * sk / math.sqrt(hd)
                if floor:
                    S = (Q @ K.T) * c; S[tri] = -np.inf
                    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
                    out[b, h] = torch.from_numpy(P @ (V * sv)).float(); continue
                mult, shift = scale_cfg(c)
                O = np.zeros((T, hd))
                for qb in range(T // N):
                    cfg = dict(ntiles=T // N, causal=1, qblk=qb, sm_mult=mult, sm_shift=shift,
                               th_qk=th_qk, th_pv=th_pv)
                    m = tiled_model(Q[qb * N:(qb + 1) * N], K, V, cfg, N, MAXT)
                    O[qb * N:(qb + 1) * N] = m['O'] / 2 ** 8 * sv
                    if st is not None:
                        st.mac += m['mac_qk'] + m['mac_pv']; st.mac_dense += 2 * m['te'] * N ** 3
                        st.tiles += m['te']; st.tiles_dense += T // N; st.cycles += LAT[m['te']]
                        st.q_zero += int((Q[qb * N:(qb + 1) * N] == 0).sum()); st.q_tot += N * N
                        st.p_zero += m['p_zero']; st.p_tot += m['p_valid']
                if st is not None:
                    S = Qf @ Kf.T / math.sqrt(hd); S[tri] = -np.inf
                    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
                    Oref = P @ Vf
                    st.err.append((layer, float(np.linalg.norm(O - Oref) / (np.linalg.norm(Oref) + 1e-12))))
                    st.heads += 1
                out[b, h] = torch.from_numpy(O).float()
        return out
    return fn


def token_losses(model, X, Y, make_override, bs=64):
    for i, blk in enumerate(model.blocks): blk.a.override = make_override(i) if make_override else None
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits, _ = model(X[i:i + bs])
            out.append(F.cross_entropy(logits.transpose(1, 2), Y[i:i + bs], reduction='none').numpy())
    for blk in model.blocks: blk.a.override = None
    return np.concatenate(out)


def paired_ci(a, b, rng, n_boot=2000):
    d = (a - b).mean(axis=1)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return dict(delta=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def main():
    global N, MAXT, LAT
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('corpus')
    ap.add_argument('--nseq', type=int, default=512); ap.add_argument('--tag', default='hd8')
    a = ap.parse_args()
    ck = torch.load(a.ckpt, weights_only=False)
    cfgm = ck['cfg']; T = cfgm['ctx']
    N = cfgm['d_model'] // cfgm['n_head']; MAXT = T // N
    rt = json.load(open(os.path.join(ROOT, 'results', 'regression_tiled.json')))
    LAT = {int(k): v[0] for r in rt if r['N'] == N and r['MAXT'] == MAXT for k, v in r['latency_by_tiles'].items()}
    assert LAT, f'no RTL latency table for N={N}, MAXT={MAXT}'
    _, va, chars = load_corpus(a.corpus)
    model = GPT(len(chars), **cfgm); model.load_state_dict(ck['state']); model.eval()
    rng = np.random.default_rng(123)
    ix = rng.integers(0, len(va) - T - 1, a.nseq)
    X = torch.from_numpy(np.stack([va[i:i + T] for i in ix])); Y = torch.from_numpy(np.stack([va[i + 1:i + T + 1] for i in ix]))
    t0 = time.time(); brng = np.random.default_rng(7)
    res = dict(model=cfgm, head_dim=N, key_tiles=MAXT, nseq=a.nseq, tokens=a.nseq * T)
    L_fp = token_losses(model, X, Y, None)
    L_fl = token_losses(model, X, Y, lambda i: hw_attention(floor=True))
    st = Stats()
    L_hw = token_losses(model, X, Y, lambda i: hw_attention(st=st, layer=i))
    res['loss_fp32'] = float(L_fp.mean()); res['loss_int8_floor'] = float(L_fl.mean()); res['loss_hw'] = float(L_hw.mean())
    res['delta_floor_vs_fp32'] = paired_ci(L_fl, L_fp, brng)
    res['delta_hw_vs_fp32'] = paired_ci(L_hw, L_fp, brng)
    res['delta_hw_vs_floor'] = paired_ci(L_hw, L_fl, brng)
    nl = cfgm['n_layer']; errs = np.array([e for _, e in st.err]); lay = np.array([l for l, _ in st.err])
    res['hw'] = dict(mac_frac=st.mac / st.mac_dense, tiles_frac=st.tiles / st.tiles_dense,
                     mac_vs_noncausal_dense=st.mac / (2 * st.tiles_dense * N ** 3),
                     q_zero=st.q_zero / st.q_tot, p_zero=st.p_zero / st.p_tot, heads=st.heads,
                     cycles_per_head=st.cycles / st.heads, cycles_per_head_no_skip=(T // N) * LAT[T // N],
                     head_err_mean=float(errs.mean()), head_err_p95=float(np.percentile(errs, 95)),
                     head_err_by_layer=[float(errs[lay == i].mean()) for i in range(nl)])
    print(json.dumps(res, indent=1), f'({time.time() - t0:.0f}s)', flush=True)
    sweep = []
    for th_qk, th_pv in ((0, 2), (0, 4), (0, 8), (2, 0), (4, 0), (4, 4)):
        st = Stats()
        L = token_losses(model, X, Y, lambda i: hw_attention(th_qk, th_pv, st, layer=i))
        sweep.append(dict(th_qk=th_qk, th_pv=th_pv, loss=float(L.mean()), mac_frac=st.mac / st.mac_dense,
                          vs_hw=paired_ci(L, L_hw, brng), vs_fp32=paired_ci(L, L_fp, brng)))
        print(sweep[-1], f'({time.time() - t0:.0f}s)', flush=True)
    res['gating_sweep'] = [dict(th_qk=0, th_pv=0, loss=res['loss_hw'], mac_frac=res['hw']['mac_frac'],
                                vs_hw=dict(delta=0.0, ci95=[0.0, 0.0]), vs_fp32=res['delta_hw_vs_fp32'])] + sweep
    json.dump(res, open(os.path.join(ROOT, 'results', f'gpt_characterization_{a.tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
