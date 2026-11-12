#!/usr/bin/env python3
import sys, time, os
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('THREADS', '1')))
from common import *

FT_STEPS, BS, FT_LR = 400, 32, float(os.environ.get('FT_LR', '3e-4'))
X, Y = eval_set(512)


def ft(m, **kw):
    return finetune(m, FT_STEPS, lr=FT_LR, bs=BS, **kw)


def ctrl_losses():
    f = os.path.join(RES, 'control_losses.npy')
    if not os.path.exists(f):
        raise SystemExit('run "experiments.py control" first')
    return np.load(f)


def base_losses():
    f = os.path.join(RES, 'base_losses.npy')
    if not os.path.exists(f):
        m, _ = load(); np.save(f, token_losses(m, X, Y))
    return np.load(f)


def summarize(m, L, extra=None, hw=False):
    Lc = ctrl_losses()
    r = dict(loss=float(L.mean()), vs_control=paired(L, Lc), vs_base=paired(L, base_losses()),
             linear_sparsity=sparsity_of(m), attn_cycles=attention_cycles(m))
    if hw:
        Lh = hw_eval(m, X, Y)
        r['hw'] = dict(loss=float(Lh.mean()), hw_vs_fp32_same_model=paired(Lh, L), hw_vs_control_fp32=paired(Lh, Lc))
    if extra: r.update(extra)
    return r


def exp_control():
    m, cfg = load(); Lb = base_losses()
    ft(m); L = token_losses(m, X, Y); np.save(os.path.join(RES, 'control_losses.npy'), L)
    torch.save(m.state_dict(), os.path.join(RES, 'control.pt'))
    Lh = hw_eval(m, X, Y)
    save('control', dict(loss_base=float(Lb.mean()), loss_control=float(L.mean()), control_vs_base=paired(L, Lb),
                         attn_cycles=attention_cycles(m), hw_loss=float(Lh.mean()), hw_vs_fp32=paired(Lh, L),
                         recipe=dict(steps=FT_STEPS, batch=BS, lr=FT_LR)))


def exp_magnitude():
    out = []
    for sp in (0.3, 0.5, 0.7, 0.8, 0.9):
        m, cfg = load()
        masks = magnitude_masks(m, sp); apply_masks(masks)
        r_one = summarize(m, token_losses(m, X, Y))
        ft(m, masks=masks)
        r_ft = summarize(m, token_losses(m, X, Y))
        g, _ = load()
        state = {'masks': None}
        ramp = int(0.75 * FT_STEPS)
        def cb(step, g=g, sp=sp, state=state):
            if step % 25 == 0 and step <= ramp:
                s_t = sp * (1 - (1 - step / ramp) ** 3)
                state['masks'] = magnitude_masks(g, s_t) if s_t > 0 else None
            if state['masks']: apply_masks(state['masks'])
        finetune(g, FT_STEPS, lr=FT_LR, bs=BS, step_cb=cb)
        apply_masks(magnitude_masks(g, sp))
        r_gr = summarize(g, token_losses(g, X, Y))
        out.append(dict(sparsity=sp, one_shot=r_one, one_shot_ft=r_ft, gradual_ft=r_gr))
        print(sp, round(r_one['loss'], 4), round(r_ft['loss'], 4), round(r_gr['loss'], 4), flush=True)
    save('magnitude', out)


def head_importance(m, nbatch=40):
    d = data(); rng = np.random.default_rng(1)
    gates = [torch.ones(a.h, requires_grad=True) for a in attns(m)]
    for a, g in zip(attns(m), gates): a.head_mask = g
    imp = [torch.zeros(a.h) for a in attns(m)]
    m.eval()
    for _ in range(nbatch):
        x, y = batch(d['tr'], 32, 64, rng)
        _, loss = m(x, y)
        gr = torch.autograd.grad(loss, gates)
        for i, gg in enumerate(gr): imp[i] += gg.abs()
    for a in attns(m): a.head_mask = None
    return [(t / (t.norm() + 1e-12)) for t in imp]


def prune_heads(m, imp, k, rng=None):
    H = attns(m)[0].h
    flat = [(float(imp[l][h]) if rng is None else float(rng.random()), l, h) for l in range(len(imp)) for h in range(H)]
    flat.sort()
    masks = [torch.ones(H) for _ in imp]; removed = 0
    for _, l, h in flat:
        if removed == k: break
        if masks[l].sum() > 1:
            masks[l][h] = 0.0; removed += 1
    for a, mk in zip(attns(m), masks): a.head_mask = mk
    return [mk.tolist() for mk in masks]


def exp_heads():
    base, cfg = load(); imp = head_importance(base)
    out = dict(importance=[t.tolist() for t in imp], runs=[])
    for k in (4, 8, 12, 16, 20, 24):
        m, _ = load(); mk = prune_heads(m, imp, k)
        r_one = summarize(m, token_losses(m, X, Y))
        ft(m); r_ft = summarize(m, token_losses(m, X, Y), hw=(k in (8, 16)))
        out['runs'].append(dict(pruned=k, masks=mk, one_shot=r_one, ft=r_ft))
        print(k, round(r_one['loss'], 4), round(r_ft['loss'], 4), r_ft['attn_cycles'], flush=True)
    for k in (8, 16):
        m, _ = load(); prune_heads(m, imp, k, rng=np.random.default_rng(k))
        r_one = summarize(m, token_losses(m, X, Y)); ft(m)
        out['runs'].append(dict(pruned=k, random=True, one_shot=r_one, ft=summarize(m, token_losses(m, X, Y))))
    save('heads', out)


def exp_nm():
    out = []
    for n, mm in ((2, 4), (1, 4)):
        m, _ = load(); masks = nm_masks(m, n, mm); apply_masks(masks)
        r_one = summarize(m, token_losses(m, X, Y)); ft(m, masks=masks)
        out.append(dict(pattern=f'{n}:{mm}', one_shot=r_one, ft=summarize(m, token_losses(m, X, Y))))
        print(n, mm, round(r_one['loss'], 4), round(out[-1]['ft']['loss'], 4), flush=True)
    save('nm', out)


def exp_quant():
    out = dict(ptq=[], qat=[])
    for bits, grp in ((8, 0), (6, 0), (4, 0), (4, 16), (3, 16)):
        m, _ = load(); add_weight_quant(m, bits, grp)
        out['ptq'].append(dict(bits=bits, group=grp, **summarize(m, token_losses(m, X, Y))))
        print('ptq', bits, grp, round(out['ptq'][-1]['loss'], 4), flush=True)
    for bits, grp in ((4, 0), (4, 16), (3, 16)):
        m, _ = load(); add_weight_quant(m, bits, grp); ft(m)
        out['qat'].append(dict(bits=bits, group=grp, **summarize(m, token_losses(m, X, Y))))
        print('qat', bits, grp, round(out['qat'][-1]['loss'], 4), flush=True)
    m, _ = load(); set_attn(m, qat_qkv=True); ft(m); set_attn(m, qat_qkv=False)
    L = token_losses(m, X, Y)
    out['attn_qat'] = summarize(m, L, hw=True)
    print('attn qat', out['attn_qat']['hw'], flush=True)
    save('quant', out)


def exp_distill(steps=3000):
    teacher, cfg = load()
    scfg = dict(cfg, n_layer=2)
    out = dict(student_cfg=scfg, steps=steps)
    for name, t in (('scratch', None), ('kd', teacher)):
        torch.manual_seed(0); s = GPT(len(data()['chars']), **scfg)
        t0 = time.time(); finetune(s, steps, lr=3e-3, bs=BS, teacher=t, kd_alpha=0.7, kd_T=2.0)
        L = token_losses(s, X, Y)
        out[name] = dict(loss=float(L.mean()), vs_control=paired(L, ctrl_losses()),
                         params=sum(p.numel() for p in s.parameters()), attn_cycles=attention_cycles(s),
                         train_s=round(time.time() - t0))
        np.save(os.path.join(RES, f'distill_{name}_losses.npy'), L)
        print(name, round(out[name]['loss'], 4), flush=True)
    out['kd_vs_scratch'] = paired(np.load(os.path.join(RES, 'distill_kd_losses.npy')),
                                  np.load(os.path.join(RES, 'distill_scratch_losses.npy')))
    out['teacher_params'] = sum(p.numel() for p in teacher.parameters())
    save('distill', out)


def exp_window():
    out = []
    for wt in (1, 2, 3, 4):
        m, _ = load(); set_attn(m, window_tiles=wt)
        r_zero = summarize(m, token_losses(m, X, Y))
        ft(m); r_ft = summarize(m, token_losses(m, X, Y), hw=(wt in (2, 3)))
        out.append(dict(window_tiles=wt, window_tokens=8 * wt, zero_shot=r_zero, ft=r_ft))
        print(wt, round(r_zero['loss'], 4), round(r_ft['loss'], 4), r_ft['attn_cycles'], flush=True)
    save('window', out)


def exp_combined():
    base, cfg = load(); imp = head_importance(base)
    plans = [dict(name='naive: -8 heads, window 24, 2:4 weights', k=8, wt=3, nm=True, wq=None),
             dict(name='naive: -12 heads, window 16, 2:4 weights', k=12, wt=2, nm=True, wq=None),
             dict(name='informed: window 24, INT4/g16 QAT', k=0, wt=3, nm=False, wq=(4, 16)),
             dict(name='informed: window 16, INT4/g16 QAT', k=0, wt=2, nm=False, wq=(4, 16)),
             dict(name='informed: -4 heads, window 24, INT4/g16 QAT', k=4, wt=3, nm=False, wq=(4, 16))]
    out = []
    for p in plans:
        m, _ = load()
        if p['k']: prune_heads(m, imp, p['k'])
        set_attn(m, window_tiles=p['wt'], qat_qkv=True)
        masks = None
        if p['nm']: masks = nm_masks(m, 2, 4); apply_masks(masks)
        if p['wq']: add_weight_quant(m, *p['wq'])
        ft(m, masks=masks); set_attn(m, qat_qkv=False)
        r = summarize(m, token_losses(m, X, Y), hw=True, extra=dict(plan=p['name'], pruned_heads=p['k'],
                      window_tiles=p['wt'], weights='2:4' if p['nm'] else f'INT{p["wq"][0]}/g{p["wq"][1]}'))
        out.append(r); print(p['name'], round(r['loss'], 4), r['attn_cycles'], round(r['hw']['loss'], 4), flush=True)
    save('combined', out)


if __name__ == '__main__':
    t0 = time.time()
    globals()['exp_' + sys.argv[1]]()
    print(f'done {sys.argv[1]} in {time.time() - t0:.0f}s')
