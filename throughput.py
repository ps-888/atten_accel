import json, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
N, MAXT = 8, 8
rt = json.load(open(os.path.join(ROOT, 'results', 'regression_tiled.json')))
lat = {int(k): v[0] for r in rt if r['N'] == N and r['MAXT'] == MAXT for k, v in r['latency_by_tiles'].items()}
sta = open(os.path.join(ROOT, 'results', 'syn', 'sta_tiled8d.txt')).read()
period_ns = 2.0
slack = float(re.search(r'worst slack max (-?[0-9.]+)', sta).group(1))
fmax_mhz = 1000.0 / (period_ns - slack)
util_8 = 2 * N * MAXT / lat[MAXT]
tiles_causal = sum(qb + 1 for qb in range(MAXT))
cycles_causal = sum(lat[qb + 1] for qb in range(MAXT))
ops_causal = tiles_causal * 2 * N ** 3 * 2
ops_per_cycle = ops_causal / cycles_causal
res = dict(
    fmax_mhz=round(fmax_mhz, 1),
    array_utilization_64_tokens=round(util_8, 3),
    causal_head_64_tokens=dict(tiles=tiles_causal, cycles=cycles_causal, ops=ops_causal,
                               ops_per_cycle=round(ops_per_cycle, 1),
                               effective_gops=round(ops_per_cycle * fmax_mhz / 1000, 1),
                               latency_us=round(cycles_causal / fmax_mhz, 2)),
    peak_gops=round(2 * N * N * fmax_mhz / 1000, 1),
)
json.dump(res, open(os.path.join(ROOT, 'results', 'throughput.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
