#!/usr/bin/env python3
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from regress_tiled import run, ROOT
out, bad = [], 0
for n, maxt, t in ((4, 4, 800), (4, 8, 800), (8, 8, 1000), (16, 4, 300)):
    r, f = run(n, maxt, t, seed=7 + n * maxt); out.append(r); bad += len(f)
json.dump(out, open(os.path.join(ROOT, 'results', 'regression_tiled.json'), 'w'), indent=1)
print('TOTAL', sum(r['tests'] for r in out), 'fails', bad); sys.exit(1 if bad else 0)
