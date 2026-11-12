#!/usr/bin/env python3
import argparse, glob, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

CFG = dict(n_layer=4, n_head=8, d_model=64, ctx=64)


def load_corpus(d):
    files = sorted(f for f in glob.glob(os.path.join(d, '*.txt')))
    text = '\n'.join(open(f, encoding='latin-1').read() for f in files)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.95 * len(data))
    return data[:n], data[n:], chars


class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.capture = False; s.override = None
    def forward(s, x):
        B, T, C = x.shape; hd = C // s.h
        q, k, v = s.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, s.h, hd).transpose(1, 2) for t in (q, k, v))
        if s.capture: s.qkv_cache = (q.detach(), k.detach(), v.detach())
        if s.override is not None:
            y = s.override(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.l1 = nn.LayerNorm(d); s.a = Attn(d, h); s.l2 = nn.LayerNorm(d)
        s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        x = x + s.a(s.l1(x)); return x + s.m(s.l2(x))


class GPT(nn.Module):
    def __init__(s, vocab, n_layer, n_head, d_model, ctx):
        super().__init__()
        s.tok = nn.Embedding(vocab, d_model); s.pos = nn.Embedding(ctx, d_model)
        s.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        s.ln = nn.LayerNorm(d_model); s.head = nn.Linear(d_model, vocab, bias=False)
    def forward(s, idx, tgt=None):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        logits = s.head(s.ln(x))
        loss = None if tgt is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        return logits, loss


def batch(data, bs, ctx, rng):
    ix = rng.integers(0, len(data) - ctx - 1, bs)
    x = torch.from_numpy(np.stack([data[i:i + ctx] for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + ctx + 1] for i in ix]))
    return x, y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('corpus'); ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--n_head', type=int, default=CFG['n_head'])
    ap.add_argument('--d_model', type=int, default=CFG['d_model']); a = ap.parse_args()
    CFG.update(n_head=a.n_head, d_model=a.d_model)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    tr, va, chars = load_corpus(a.corpus)
    m = GPT(len(chars), **CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(a.steps + 1):
        lr = 3e-3 * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups: g['lr'] = lr
        x, y = batch(tr, 64, CFG['ctx'], rng)
        _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                vl = np.mean([m(*batch(va, 64, CFG['ctx'], rng))[1].item() for _ in range(10)])
            print(f'step {step} train {loss.item():.3f} val {vl:.3f} ({time.time() - t0:.0f}s)', flush=True)
    torch.save(dict(state=m.state_dict(), chars=chars, cfg=CFG), a.out)


if __name__ == '__main__':
    main()
