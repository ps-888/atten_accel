import json, math, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.parametrize as P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'workload')); sys.path.insert(0, os.path.join(ROOT, 'model'))
from train_tiny_gpt import GPT, Attn, load_corpus, batch
from attn_model import tiled_model, HP

CORPUS = os.environ.get('CORPUS', os.path.join(ROOT, 'build', 'tools', 'gutenberg'))
RES = os.path.join(ROOT, 'compress', 'results'); os.makedirs(RES, exist_ok=True)
_DATA = {}


def data():
    if not _DATA:
        tr, va, chars = load_corpus(CORPUS)
        _DATA.update(tr=tr, va=va, chars=chars)
    return _DATA


def fq_int8_dyn(x):
    s = x.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(x / s), -128, 127) * s
    return x + (q - x).detach()


def window_mask(T, tile, wt, device=None):
    i = torch.arange(T, device=device)[:, None]; j = torch.arange(T, device=device)[None, :]
    return (j <= i) & ((j // tile) >= (i // tile) - (wt - 1))


def attn_forward(s, x):
    B, T, C = x.shape; hd = C // s.h
    q, k, v = s.qkv(x).split(C, dim=2)
    q, k, v = (t.view(B, T, s.h, hd).transpose(1, 2) for t in (q, k, v))
    if getattr(s, 'qat_qkv', False):
        q, k, v = fq_int8_dyn(q), fq_int8_dyn(k), fq_int8_dyn(v)
    if getattr(s, 'capture', False): s.qkv_cache = (q.detach(), k.detach(), v.detach())
    if getattr(s, 'override', None) is not None:
        y = s.override(q, k, v)
    elif getattr(s, 'window_tiles', None):
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(T, hd, s.window_tiles, x.device))
    else:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    hm = getattr(s, 'head_mask', None)
    if hm is not None:
        y = y * hm.view(1, -1, 1, 1)
    return s.o(y.transpose(1, 2).reshape(B, T, C))


Attn.forward = attn_forward


def attns(model):
    return [b.a for b in model.blocks]


def set_attn(model, **kw):
    for a in attns(model):
        for k, v in kw.items():
            setattr(a, k, v)


def load(path=os.path.join(ROOT, 'workload', 'tiny_gpt.pt')):
    ck = torch.load(path, weights_only=False)
    m = GPT(len(ck['chars']), **ck['cfg']); m.load_state_dict(ck['state'])
    return m, ck['cfg']


def clone(model, cfg):
    m = GPT(len(data()['chars']), **cfg); m.load_state_dict(model.state_dict())
    for a_src, a_dst in zip(attns(model), attns(m)):
        for k in ('head_mask', 'window_tiles', 'qat_qkv'):
            if hasattr(a_src, k):
                v = getattr(a_src, k); setattr(a_dst, k, v.clone() if torch.is_tensor(v) else v)
    return m


def linears(model):
    out = []
    for i, b in enumerate(model.blocks):
        out += [(f'b{i}.qkv', b.a.qkv), (f'b{i}.o', b.a.o), (f'b{i}.fc1', b.m[0]), (f'b{i}.fc2', b.m[2])]
    return out


def n_linear_params(model):
    return sum(l.weight.numel() for _, l in linears(model))


def apply_masks(masks):
    with torch.no_grad():
        for lin, m in masks:
            lin.weight.mul_(m)


def magnitude_masks(model, sparsity):
    W = torch.cat([l.weight.detach().abs().flatten() for _, l in linears(model)])
    k = int(sparsity * W.numel())
    thr = torch.kthvalue(W, k).values if k > 0 else torch.tensor(-1.0)
    return [(l, (l.weight.detach().abs() > thr).float()) for _, l in linears(model)]


def nm_masks(model, n=2, m=4):
    out = []
    for _, l in linears(model):
        W = l.weight.detach().abs(); O, I = W.shape
        g = W.view(O, I // m, m)
        idx = g.topk(n, dim=-1).indices
        mk = torch.zeros_like(g).scatter_(-1, idx, 1.0).view(O, I)
        out.append((l, mk))
    return out


def sparsity_of(model):
    z = sum(int((l.weight == 0).sum()) for _, l in linears(model))
    return z / n_linear_params(model)


class FakeQuant(nn.Module):
    def __init__(s, bits, group=0):
        super().__init__(); s.bits, s.group = bits, group
    def forward(s, w):
        qmax = 2 ** (s.bits - 1) - 1
        O, I = w.shape
        g = s.group or I
        wg = w.reshape(O, I // g, g)
        sc = wg.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        q = (torch.clamp(torch.round(wg / sc), -qmax - 1, qmax) * sc).reshape(O, I)
        return w + (q - w).detach()


def add_weight_quant(model, bits, group=0):
    for _, l in linears(model):
        P.register_parametrization(l, 'weight', FakeQuant(bits, group))


def bake_weight_quant(model):
    for _, l in linears(model):
        if P.is_parametrized(l, 'weight'):
            P.remove_parametrizations(l, 'weight', leave_parametrized=True)


def finetune(model, steps, lr=1e-3, masks=None, teacher=None, kd_alpha=0.5, kd_T=2.0, seed=0,
             bs=64, step_cb=None, log_every=0):
    d = data(); rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ctx = model.pos.num_embeddings
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    model.train()
    if teacher is not None: teacher.eval()
    for step in range(steps):
        for g in opt.param_groups: g['lr'] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        if step_cb is not None: step_cb(step)
        x, y = batch(d['tr'], bs, ctx, rng)
        logits, ce = model(x, y)
        loss = ce
        if teacher is not None:
            with torch.no_grad(): tl, _ = teacher(x)
            kd = F.kl_div(F.log_softmax(logits / kd_T, -1), F.log_softmax(tl / kd_T, -1),
                          log_target=True, reduction='batchmean') * kd_T ** 2 / x.shape[1]
            loss = (1 - kd_alpha) * ce + kd_alpha * kd
        opt.zero_grad(); loss.backward(); opt.step()
        if masks: apply_masks(masks)
        if log_every and step % log_every == 0: print(f'  step {step} loss {ce.item():.3f}', flush=True)
    model.eval()
    return model


def eval_set(nseq=512, T=64):
    d = data(); rng = np.random.default_rng(123)
    ix = rng.integers(0, len(d['va']) - T - 1, nseq)
    X = torch.from_numpy(np.stack([d['va'][i:i + T] for i in ix]))
    Y = torch.from_numpy(np.stack([d['va'][i + 1:i + T + 1] for i in ix]))
    return X, Y


def token_losses(model, X, Y, bs=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg, _ = model(X[i:i + bs])
            out.append(F.cross_entropy(lg.transpose(1, 2), Y[i:i + bs], reduction='none').numpy())
    return np.concatenate(out)


def paired(a, b, n_boot=2000, seed=7):
    d = (a - b).mean(axis=1); rng = np.random.default_rng(seed)
    bo = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return dict(delta=float(d.mean()), ci95=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))])


def _lat_table(n=8, maxt=8):
    rt = json.load(open(os.path.join(ROOT, 'results', 'regression_tiled.json')))
    return {int(k): v[0] for r in rt if r['N'] == n and r['MAXT'] == maxt for k, v in r['latency_by_tiles'].items()}


def _quant(x):
    s = max(float(np.abs(x).max()), 1e-12) / 127.0
    return np.clip(np.round(x / s), -128, 127).astype(np.int64), s


def _scale_cfg(c):
    c2 = c * math.log2(math.e) * 2 ** HP['XF']
    shift = int(min(63, max(0, math.floor(math.log2(65535 / c2)))))
    return int(min(65535, max(1, round(c2 * 2 ** shift)))), shift


def head_cycles(T=64, n=8, window_tiles=None):
    lat = _lat_table(n, T // n); tot = 0
    for qb in range(T // n):
        hi = qb + 1; lo = max(0, qb - window_tiles + 1) if window_tiles else 0
        tot += lat[hi - lo]
    return tot


def attention_cycles(model, T=64):
    tot = 0
    for a in attns(model):
        live = int(a.head_mask.sum()) if getattr(a, 'head_mask', None) is not None else a.h
        tot += live * head_cycles(T, a.qkv.in_features // a.h, getattr(a, 'window_tiles', None))
    return tot


def hw_eval(model, X, Y):
    N = MAXT = None
    def make(a):
        def fn(q, k, v):
            B, H, T, hd = q.shape; n = hd; mt = T // n
            out = torch.zeros_like(q)
            hm = getattr(a, 'head_mask', None); wt = getattr(a, 'window_tiles', None)
            for b in range(B):
                for h in range(H):
                    if hm is not None and float(hm[h]) == 0.0:
                        continue
                    Q, sq = _quant(q[b, h].double().numpy()); K, sk = _quant(k[b, h].double().numpy())
                    V, sv = _quant(v[b, h].double().numpy())
                    mult, shift = _scale_cfg(sq * sk / math.sqrt(hd))
                    O = np.zeros((T, hd))
                    for qb in range(mt):
                        cfg = dict(ntiles=mt, causal=1, qblk=qb, sm_mult=mult, sm_shift=shift, th_qk=0, th_pv=0,
                                   tile_lo=max(0, qb - wt + 1) if wt else 0)
                        O[qb * n:(qb + 1) * n] = tiled_model(Q[qb * n:(qb + 1) * n], K, V, cfg, n, mt)['O'] / 256.0 * sv
                    out[b, h] = torch.from_numpy(O).float()
            return out
        return fn
    for a in attns(model): a.override = make(a)
    try:
        L = token_losses(model, X, Y, bs=32)
    finally:
        for a in attns(model): a.override = None
    return L


def save(name, obj):
    json.dump(obj, open(os.path.join(RES, f'{name}.json'), 'w'), indent=1)
