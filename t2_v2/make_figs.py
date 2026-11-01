"""Figures for the track 2 PR (22.9 s -> 15.9 s), one message each.

    /home/chunyuan/noise_lane/plotenv/bin/python make_figs.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
LOGS = Path('/home/chunyuan/noise_t2/logs')
RECORD = ('/home/chunyuan/noise_t2', 'ebee4e3', 'train/speedrun/log')  # #52's official log in upstream main
OFFICIAL = Path('/tmp/claude-chunyuan/port/wt/logs/speedrun')  # this PR's README-exact run
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GREY, INK, MUTED, GRID = '#8c8c8c', '#222222', '#666666', '#ececec'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b0b0b0', 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'legend.frameon': False})


def git_json(path, name):
    text = subprocess.run(['git', '-C', RECORD[0], 'show', f'{RECORD[1]}:{path}/{name}'], check=True,
                          capture_output=True, text=True).stdout
    return text


def rows_of(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# 1. each change on top of the previous ones (seed 0, our node; times of clean runs)
SPEED, BUY, SELL = BLUE, ORANGE, AQUA
steps = [
    ('#52, our node (5 runs)', GREY, 22.70, 3.4998),
    ('+ same-number speedups', SPEED, 22.21, 3.4992),
    ('+ init and merge constants', BUY, 22.33, 3.4965),
    ('+ root 4 on half the tokens, root 6 as\n   its control, importance draw, hand-off 250', SELL, 20.27, 3.4964),
    ('+ overhead in CUDA graphs, root 6 along\n   its routed span, partner gain, fast weight', SPEED, 19.26, 3.4908),
    ('+ 16 rows per update from half the horizon', SPEED, 17.96, 3.4945),
    ('+ hand-off 175, root 4 on 3/8', SELL, 17.35, 3.4950),
    ('+ root 4 on 3/16, then 7/16', SELL, 17.06, 3.4957),
    ('+ fixed merge rule', SPEED, 16.92, 3.4957),
    ('the same, timed later that day (5 runs)', GREY, 17.04, 3.4961),
    ('+ chain kernels with fewer instructions', SPEED, 16.79, 3.4954),
    ('+ one-pass moment-query kernel', SPEED, 16.58, 3.4960),
    ('+ difficulty order on the GPU', SPEED, 16.53, 3.4965),
    ('+ word-mix pruning of 1.5% of the rows', SELL, 16.30, 3.4960),
    ('+ key-scan and value-credit kernels (this PR)', SPEED, 15.96, 3.4962),
]
fig, ax = plt.subplots(figsize=(8.6, 7.0))
y = list(range(len(steps)))[::-1]
for k, (yi, (label, color, t, v)) in enumerate(zip(y, steps)):
    ax.barh(yi, t, height=0.62, color=color)
    dt = '' if k in (0, 9) else f'  ({t - steps[k - 1][2]:+.2f} s)'.replace('-', '−')
    ax.text(t + 0.15, yi, f'{t:.2f} s{dt}', va='center', fontsize=10, color=INK)
    ax.text(1.005, yi, f'{v:.4f}', transform=ax.get_yaxis_transform(), va='center', fontsize=10, color=MUTED)
ax.text(1.005, 1.01, 'val\nloss', transform=ax.transAxes, va='bottom', fontsize=9, color=MUTED, linespacing=1.1)
ax.axhline(len(steps) - 9.5, color='#c8c8c8', lw=1.0, xmin=-0.6, clip_on=False)  # the later day's block below
ax.set_yticks(y, [s[0] for s in steps], fontsize=10)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 34)
ax.set_ylim(-0.55, len(steps) - 0.45)
ax.set_xlabel('training time on 8xH100 (s), seed 0')
handles = [Patch(color=SPEED, label='faster, same numbers'), Patch(color=SELL, label='cheaper estimate (sells loss)'),
           Patch(color=BUY, label='more learning (buys loss)')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.012, 0.94), ncol=3, fontsize=9.5, handlelength=1.2,
           columnspacing=1.2)
fig.text(0.02, 0.012, 'Research runs on our node; every row stays below 3.5. The lower block was timed later the same day, when the node ran\n'
         '0.12 s slower (interleaved runs, means of 2-5). The official run of this PR: 3.4959 in 15.91 s.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Each change, on top of the previous ones', x=0.02, y=0.975, ha='left', fontsize=13.5)
fig.subplots_adjust(left=0.40, right=0.90, top=0.87, bottom=0.155)
fig.savefig(OUT / 'fig1_steps.png', dpi=200)


# 2. train loss along training time: #52's official log vs this PR's run
def curve(rows, every=16):
    st = [r for r in rows if 'step' in r and 'train_seconds' in r and r.get('phase') == 'cut']
    out = []
    for i in range(every, len(st) + 1, every):
        chunk = st[i - every:i]
        out.append((chunk[-1]['train_seconds'], sum(r['train_loss'] for r in chunk) / len(chunk)))
    return out


rec_rows = rows_of(git_json(RECORD[2], 'metrics.jsonl'))
rec_res = json.loads(git_json(RECORD[2], 'result.json'))
pr_dir = OFFICIAL
pr_res = json.loads((pr_dir / 'result.json').read_text())  # the official result
tele = LOGS / 'fig_v2_tele16'  # same script, flushing every 16 updates, for the time stamps along the run
pr_rows = rows_of((tele / 'metrics.jsonl').read_text()) if (tele / 'metrics.jsonl').exists() else []
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color, rows, res in [('#52', GREY, rec_rows, rec_res), ('this PR', BLUE, pr_rows, pr_res)]:
    pts = curve(rows)
    if pts:
        ax.plot(*zip(*pts), color=color, lw=2, label=label)
    ax.scatter([res['train_seconds']], [res['val_loss']], color=color, s=45, zorder=3, edgecolor='white', linewidth=1.2)
    ax.annotate(f"{res['val_loss']:.4f}\nin {res['train_seconds']:.1f} s", (res['train_seconds'], res['val_loss']),
                textcoords='offset points', xytext=(-14, -30), fontsize=10, color=INK, ha='center')
ax.axhline(3.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.text(0.3, 3.5, 'target 3.5', va='bottom', fontsize=9, color=MUTED)
ax.set_xlim(0, 25)
ax.set_ylim(3.35, 4.3)
ax.set_xlabel('training time on 8xH100 (s)')
ax.set_ylabel('train loss (Cut updates, mean of 16)')
ax.grid(axis='y', color=GRID, lw=0.8)
ax.legend(loc='upper right', fontsize=11)
fig.text(0.02, 0.015, 'Seed 0. Dots: final validation loss after the end merge (official logs).\nThis PR\'s curve '
         'comes from the same script flushing its records every 16 updates.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Lower loss sooner', x=0.02, ha='left', fontsize=13.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / 'fig2_curves.png', dpi=200)
print('ok', pr_dir)
