import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
D = os.path.join(ROOT, 'docs'); os.makedirs(D, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a85'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e3', 'axes.axisbelow': True, 'axes.edgecolor': '#b5b5b0', 'axes.titleweight': 'bold'})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(D, name), dpi=150); plt.close(fig)


RT = json.load(open(os.path.join(ROOT, 'results', 'regression_tiled.json')))
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for col, (n, mt) in zip((AQUA, BLUE, ORANGE), ((4, 8), (8, 8), (16, 4))):
    r = [x for x in RT if x['N'] == n and x['MAXT'] == mt][0]['latency_by_tiles']
    ts = sorted(int(k) for k in r)
    ax.plot(ts, [100 * 2 * n * t / r[str(t)][0] for t in ts], '-o', color=col, lw=2, ms=4, label=f'N={n}')
ax.set_xlabel('key tiles processed (sequence length / N)'); ax.set_ylabel('array utilization (%)')
ax.set_ylim(0, 100); ax.set_title('MAC-array utilization (RTL)'); ax.legend(frameon=False)
save(fig, 'tiled_utilization.png')

GP = [(t, os.path.join(ROOT, 'results', f'gpt_characterization_{t}.json')) for t in ('hd8', 'hd16')]
GP = [(t, json.load(open(f))) for t, f in GP if os.path.exists(f)]
if GP:
    fig, axs = plt.subplots(1, len(GP), figsize=(5.2 * len(GP), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (tag, g) in zip(axs, GP):
        fl = g['delta_floor_vs_fp32']
        ax.axhspan(1e3 * fl['ci95'][0], 1e3 * fl['ci95'][1], color='#e6e6e3', zorder=0)
        ax.axhline(1e3 * fl['delta'], color=GRAY, lw=1, ls='--')
        ax.text(0.01, 1e3 * fl['delta'], ' INT8 floor (band = 95% CI)', transform=ax.get_yaxis_transform(),
                ha='left', va='bottom', fontsize=8, color='#555')
        ax.axhline(0, color='#b5b5b0', lw=1)
        for d in g['gating_sweep']:
            v = d['vs_fp32']; col = BLUE if (d['th_qk'], d['th_pv']) == (0, 0) else ORANGE
            ax.errorbar(100 * d['mac_frac'], 1e3 * v['delta'],
                        yerr=[[1e3 * (v['delta'] - v['ci95'][0])], [1e3 * (v['ci95'][1] - v['delta'])]],
                        fmt='o', color=col, ms=5, capsize=3, lw=1.2)
            ax.annotate(f"({d['th_qk']},{d['th_pv']})", (100 * d['mac_frac'], 1e3 * v['delta']),
                        textcoords='offset points', xytext=(5, 2), fontsize=7)
        ax.set_title(f"head_dim {g['head_dim']} ({g['tokens'] // 1000}k tokens)")
        ax.set_xlabel('MACs fired (% of dense, processed tiles)')
    axs[0].set_ylabel('Δ loss vs FP32 (milli-nats/char, 95% CI)')
    fig.suptitle('Trained GPTs on the accelerator: exact (blue) vs gated (orange) (th_qk, th_pv)', fontweight='bold')
    save(fig, 'gpt_gating.png')
