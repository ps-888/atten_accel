import numpy as np

HP = dict(XF=6, EXP_FRAC=12, SEED_BITS=3, R=16, OUT_FRAC=7, T_W=32)


def exp2_lut():
    xf, ef = HP['XF'], HP['EXP_FRAC']
    return [int(round((2 ** ef) * 2 ** (-f / 2 ** xf))) for f in range(2 ** xf)]


def seed_lut():
    sb, r = HP['SEED_BITS'], HP['R']
    return [int(round((2 ** r) / (1 + (i + 0.5) / 2 ** sb))) for i in range(2 ** sb)]


_EXP, _SEED = exp2_lut(), seed_lut()


def _clog2(n):
    return max(1, (n - 1).bit_length())


def _exp2n(d):
    XF, EF = HP['XF'], HP['EXP_FRAC']
    ip = d >> XF
    return 0 if ip > EF else (_EXP[d & ((1 << XF) - 1)] >> ip)


def _gate(A, th):
    A = np.asarray(A, dtype=np.int64)
    return np.where((A > th) | (A < -th), A, 0)


def _nr(mn, y, w):
    R = HP['R']; m18 = (1 << (R + 2)) - 1
    two_m = ((2 << R) - ((mn * y >> (w - 1)) & m18)) & m18
    return ((y * two_m) >> R) & ((1 << (R + 1)) - 1)


def tiled_model(Q, K, V, cfg, n, maxt=8, of=8):
    EF, PF, SB, R, XF = HP['EXP_FRAC'], 7, HP['SEED_BITS'], HP['R'], HP['XF']
    LWID = PF + _clog2(maxt * n) + 1
    DLIM = (EF + 1) << XF
    causal, qb = cfg['causal'], cfg['qblk']
    mult, shift = cfg['sm_mult'], cfg['sm_shift']
    hi = min(cfg['ntiles'], qb + 1) if causal else cfg['ntiles']
    lo = min(cfg.get('tile_lo', 0), hi - 1)
    te = hi - lo
    m = [0] * n; mv = [False] * n; l = [0] * n
    O = np.zeros((n, n), dtype=np.int64)
    mac_qk = mac_pv = 0
    p_zero = p_valid = 0
    Qg = _gate(Q, cfg['th_qk'])
    for j in range(lo, hi):
        Kj = np.asarray(K[j * n:(j + 1) * n], dtype=np.int64)
        Vj = np.asarray(V[j * n:(j + 1) * n], dtype=np.int64)
        mac_qk += int(np.count_nonzero(Qg)) * n
        S = Qg @ Kj.T
        P = np.zeros((n, n), dtype=np.int64); alpha = [0] * n
        for r in range(n):
            srow = [int(v) for v in S[r]]
            mask = [bool(causal and (j * n + i > qb * n + r)) for i in range(n)]
            un = [srow[i] for i in range(n) if not mask[i]]
            mo, mvo = m[r], mv[r]
            mn = mo if not un else (max(un) if not mvo else max(max(un), mo))
            mvn = mvo or bool(un)
            m[r], mv[r] = mn, mvn
            s = 0
            for i in range(n):
                if mask[i] or not mvn:
                    p = 0
                else:
                    d = min(((mn - srow[i]) * mult) >> shift, DLIM)
                    e = _exp2n(d)
                    p = min((e + (1 << (EF - PF - 1))) >> (EF - PF), (1 << PF) - 1)
                P[r, i] = p; s += p
                if not mask[i]:
                    p_valid += 1; p_zero += (p == 0)
            a = _exp2n(min(((mn - mo) * mult) >> shift, DLIM)) if mvo else 0
            alpha[r] = a
            l[r] = (((l[r] * a) >> EF) + s) & ((1 << LWID) - 1)
        Pg = _gate(P, cfg['th_pv'])
        mac_pv += int(np.count_nonzero(Pg)) * n
        Rj = Pg @ Vj
        for r in range(n):
            O[r] = ((O[r] * alpha[r]) >> EF) + Rj[r]
    Of = np.zeros((n, n), dtype=np.int64)
    for r in range(n):
        lv = l[r]
        if lv == 0:
            continue
        pos = lv.bit_length() - 1
        mn = (lv << (LWID - 1 - pos)) & ((1 << LWID) - 1)
        y = _SEED[(mn >> (LWID - 1 - SB)) & ((1 << SB) - 1)]
        y = _nr(mn, _nr(mn, y, LWID), LWID)
        sh = pos + R - of
        for c in range(n):
            o = int(O[r, c]); mag = -o if o < 0 else o
            q = (mag * y + (1 << (sh - 1))) >> sh
            Of[r, c] = -q if o < 0 else q
    return dict(O=Of, l=l, Oacc=O, mac_qk=mac_qk, mac_pv=mac_pv, te=te, p_zero=p_zero, p_valid=p_valid)


def _fp_softmax_rows(L, valid=None):
    L = np.array(L, dtype=np.float64)
    if valid is not None:
        L = np.where(valid, L, -np.inf)
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def fp_reference_tiled(Q, K, V, cfg, n):
    causal, qb = cfg['causal'], cfg['qblk']
    T = min(cfg['ntiles'], qb + 1) if causal else cfg['ntiles']
    L0 = min(cfg.get('tile_lo', 0), T - 1) * n
    Kt = np.asarray(K[L0:T * n], np.float64); Vt = np.asarray(V[L0:T * n], np.float64)
    S = np.asarray(Q, np.float64) @ Kt.T
    c = cfg['sm_mult'] / 2.0 ** (cfg['sm_shift'] + HP['XF']) * np.log(2)
    valid = None
    if causal:
        valid = (np.arange(L0, T * n)[None, :] <= (qb * n + np.arange(n))[:, None])
    return _fp_softmax_rows(S * c, valid) @ Vt
