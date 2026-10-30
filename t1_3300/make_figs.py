"""Figures for the 30M track 1 PR (3.338 -> 3.300), one message each.

    /home/chunyuan/noise_lane/plotenv/bin/python make_figs.py

Values after the end merge unless noted. Train times include the end merge: the research runs leave it out of
train_seconds, so it is added back here (record #3's log already has it in).
"""
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LOGS = Path('/home/chunyuan/noise30/logs')
REPO = '/home/chunyuan/noise30_pr'
RECORD = '6f0c450ed52d52dd2fefd2a8ba485d92f2b37e92'  # record #3's commit (upstream PR #46)
PR_LOG = Path('/tmp/claude-chunyuan/d7b/pr_full/wt/train/val/log')  # this PR's official log
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'  # dataviz reference slots 1-3 (validated all-pairs on white)
GREY, INK, MUTED, GRID = '#8c8c8c', '#222222', '#666666', '#ececec'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b0b0b0', 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'legend.frameon': False})


def read(folder):
    """A log folder's result.json and per-update rows; 'record' = record #3's official log in the repo."""
    if folder == 'record':
        text = lambda name: subprocess.run(['git', '-C', REPO, 'show', f'{RECORD}:train/val/log/{name}'], check=True,
                                           capture_output=True, text=True).stdout
    else:
        text = lambda name: (Path(folder) / name).read_text()
    rows = [json.loads(line) for line in text('metrics.jsonl').splitlines() if line.strip()]
    return json.loads(text('result.json')), [r for r in rows if 'step' in r]


def minutes(result, rows):
    seconds = result['train_seconds']
    if seconds - rows[-1]['train_seconds'] < result.get('merge_seconds', 0) / 2:  # merge not timed in
        seconds += result['merge_seconds']
    return seconds / 60


def clean(rows):
    """The run had the node to itself: its Cut updates took at most 5% more than their fastest-decile pace."""
    t = [b['train_seconds'] - a['train_seconds'] for a, b in zip(rows, rows[1:]) if b['phase'] == 'cut']
    return sum(t) <= 1.05 * len(t) * sorted(t)[len(t) // 10]


def num(x, fmt):
    return format(x, fmt).replace('-', '−')


def research(name):
    result, rows = read(LOGS / name)
    return round(result['val_loss'], 4), (minutes(result, rows) if clean(rows) else None)


record, record_rows = read('record')
pr, pr_rows = read(PR_LOG)

# 1. each change on top of the previous ones (research runs, seed 0)
MEAS, INIT, SPEED = BLUE, ORANGE, AQUA
steps = [('Record #3 (research copy)', GREY, *research('d7_30m_e3_s2')),
         ('+ head transplants, self-distillation', MEAS, *research('d7_30m_he_md')),
         ('+ two-hop twin rows', MEAS, *research('d7_30m_td_kc')),
         ('+ faster code', SPEED, *research('d7_30m_kc_speed_b')),
         ('+ transit decode', MEAS, *research('d7_30m_kc_transit_gap2')),
         ('+ shared logit pairs', MEAS, *research('d7_30m_kc_transit_gap2_slq')),
         ('+ planted head, per-head polar,\n   scalar rate, linear decay', INIT, *research('d7_30m_comb_fullwd10_s0'))]
gain = {c: sum(v - steps[k - 1][2] for k, (_, cc, v, _) in enumerate(steps) if k and cc == c) for c in (MEAS, INIT, SPEED)}
fig, ax = plt.subplots(figsize=(8.4, 4.7))
lo, hi = 3.29, 3.346
y = list(range(len(steps)))[::-1]
for k, (yi, (label, color, v, m)) in enumerate(zip(y, steps)):
    ax.barh(yi, v - lo, left=lo, height=0.62, color=color)
    txt = f'{v:.4f}' if k == 0 else f'{v:.4f}  ({num(v - steps[k - 1][2], "+.4f")})'
    ax.text(v + 0.0006, yi, txt, va='center', fontsize=10, color=INK,
            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    if m is not None:
        ax.text(1.01, yi, f'{m:.0f} min', transform=ax.get_yaxis_transform(), va='center', fontsize=10, color=MUTED)
ax.text(1.01, 1.01, 'train\ntime', transform=ax.transAxes, va='bottom', fontsize=9, color=MUTED, linespacing=1.1)
ax.axvline(3.340, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0.5)
ax.text(3.3394, 0, 'backprop + ANVIL II\n(README) 3.340', ha='right', va='center', fontsize=9, color=MUTED,
        linespacing=1.1)
ax.set_yticks(y, [s for s, *_ in steps], fontsize=10.5)
ax.tick_params(axis='y', length=0)
ax.set_xticks([3.29, 3.30, 3.31, 3.32, 3.33, 3.34], ['3.29', '3.30', '3.31', '3.32', '3.33', '3.34'])
ax.set_xlim(lo, hi)
ax.set_ylim(-0.55, len(steps) - 0.45)
ax.set_xlabel('validation loss (30M, seed 0, after the end merge)')
handles = [Patch(color=MEAS, label=f'new measurements and update rules: {num(gain[MEAS], "+.4f")}'),
           Patch(color=INIT, label=f'init, optimizer, schedule: {num(gain[INIT], "+.4f")}'),
           Patch(color=SPEED, label=f'faster code: {num(gain[SPEED], "+.4f")}')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.012, 0.905), ncol=2, fontsize=9.5, handlelength=1.2,
           columnspacing=1.4)
fig.text(0.02, 0.02, f'Research runs; train time includes the end merge. Official logs: record #3 '
         f'{record["val_loss"]:.4f} in {minutes(record, record_rows):.0f} min, this PR {pr["val_loss"]:.4f} in '
         f'{minutes(pr, pr_rows):.0f} min.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Each change, on top of the previous ones', x=0.02, y=0.975, ha='left', fontsize=13.5)
fig.subplots_adjust(left=0.36, right=0.9, top=0.72, bottom=0.15)
fig.savefig(OUT / 'fig1_steps.png', dpi=200)


# 2. validation loss along training time (official logs; validation every 406 updates, untimed; the last two points
# of each line are the final weights before and after the end merge)
def curve(result, rows):
    at = {r['step']: r['train_seconds'] / 60 for r in rows}
    return ([(at[r['step']], r['val_loss']) for r in rows if 'val_loss' in r]
            + [(rows[-1]['train_seconds'] / 60, result['val_loss_plain']), (minutes(result, rows), result['val_loss'])])


lines = [('record #3', GREY, curve(record, record_rows)), ('this PR', BLUE, curve(pr, pr_rows))]
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color, pts in lines:
    ax.plot(*zip(*pts), color=color, lw=2, marker='o', ms=4.5, label=label, solid_joinstyle='round')
    ax.annotate(f'{pts[-1][1]:.3f} in {pts[-1][0]:.0f} min', pts[-1], textcoords='offset points', xytext=(9, -1),
                ha='left', va='center', fontsize=11, color=INK)
ax.set_xlim(0, 72)
ax.set_xticks(range(0, 61, 10))
ax.set_ylim(3.25, 4.5)
ax.set_xlabel('training time on 8xH100 (min)')
ax.set_ylabel('validation loss')
ax.grid(axis='y', color=GRID, lw=0.8)
ax.legend(loc='upper right', fontsize=11)
fig.text(0.02, 0.015, 'Official logs (seed 0). Validation every 406 updates, not timed.\nThe last two points of each '
         'line are the final weights before and after the end merge.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Lower loss in less time', x=0.02, ha='left', fontsize=13.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / 'fig2_curves.png', dpi=200)

# 3. the last step on five seeds (research runs; previous configuration = the row above it in fig 1)
prev = [3.3149] + [read(LOGS / f'd7_30m_base13b_s{s}')[0]['val_loss'] for s in range(1, 5)]  # seed 0's log: RESULTS.md
new = [read(LOGS / f'd7_30m_comb_fullwd10_s{s}')[0]['val_loss'] for s in range(5)]
rows3 = [(f'seed {s}', a, b) for s, (a, b) in enumerate(zip(prev, new))]
rows3.append(('mean', sum(prev) / 5, sum(new) / 5))
fig, ax = plt.subplots(figsize=(8, 3.9))
y = [5.6 - k for k in range(5)] + [0]
for yi, (label, a, b) in zip(y, rows3):
    bold = label == 'mean'
    ax.plot([a, b], [yi, yi], color='#c8c8c8', lw=2, zorder=1, solid_capstyle='round')
    ax.scatter([a], [yi], s=70 if bold else 55, color=GREY, zorder=2, edgecolor='white', linewidth=1.5)
    ax.scatter([b], [yi], s=70 if bold else 55, color=BLUE, zorder=3, edgecolor='white', linewidth=1.5)
    ax.text(1.01, yi, num(b - a, '+.4f'), transform=ax.get_yaxis_transform(), va='center', fontsize=10.5, color=INK,
            fontweight='bold' if bold else 'normal')
ax.axhline(0.8, color=GRID, lw=1)
ax.text(1.01, 6.25, 'gain', transform=ax.get_yaxis_transform(), va='bottom', fontsize=9, color=MUTED)
ax.set_yticks(y, [r[0] for r in rows3], fontsize=10.5)
for t in ax.get_yticklabels()[-1:]:
    t.set_fontweight('bold')
ax.tick_params(axis='y', length=0)
ax.set_ylim(-0.6, 6.3)
ax.set_xlim(3.297, 3.317)
ax.set_xticks([3.30, 3.305, 3.31, 3.315], ['3.300', '3.305', '3.310', '3.315'])
ax.set_xlabel('validation loss (30M, after the end merge)')
ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], color=GREY, marker='o', ls='', ms=7, label='previous configuration'),
                   Line2D([], [], color=BLUE, marker='o', ls='', ms=7, label="this PR's configuration")],
          loc='upper left', bbox_to_anchor=(-0.02, 1.2), ncol=2, fontsize=10.5, handletextpad=0.3)
fig.text(0.02, 0.015, 'Research runs: the same jobs with only --seed changed (init and probe randomness; the data order '
         'is fixed).\nSeeds 0 and 2 were used in screening.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Every seed improves', x=0.02, y=0.975, ha='left', fontsize=13.5)
fig.subplots_adjust(left=0.11, right=0.89, top=0.78, bottom=0.235)
fig.savefig(OUT / 'fig3_seeds.png', dpi=200)


# 4. the rate schedule's shape, at the same average (seeds 0 / 2)
def val2(name):
    return [read(LOGS / f'{name}_s{s}')[0]['val_loss'] for s in (0, 2)]


u = [i / 400 for i in range(401)]
scheds = [('linear to 0.3 (this PR)', BLUE, '-', [1 - 0.7 * x for x in u], val2('d7_30m_comb_fullwd10')),
          ('cosine to 0.3', GREY, (0, (1, 1.6)), [0.3 + 0.35 * (1 + math.cos(math.pi * x)) for x in u],
           val2('d7_30m_comb_fullcos')),
          ('linear to 0', GREY, (0, (5, 2, 1, 2)), [1 - x for x in u], val2('d7_30m_comb_fullwd10f0')),
          ('constant 0.65', GREY, (0, (4, 3)), [0.65 for _ in u], val2('d7_30m_comb_fullconst65'))]
fig, ax = plt.subplots(figsize=(8, 4.3))
for label, color, ls, ys, (v0, v2) in scheds:
    ax.plot(u, ys, color=color, ls=ls, lw=2.2 if color == BLUE else 1.8,
            label=f'{label}:  {v0:.4f} / {v2:.4f}')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25%', '50%', '75%', '100%'])
ax.set_yticks([0, 0.3, 0.65, 1], ['0', '0.3', '0.65', '1'])
ax.set_xlabel('training progress')
ax.set_ylabel('rate / peak rate')
ax.grid(axis='y', color=GRID, lw=0.8)
leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10.5, handlelength=2.6,
                title='schedule:  val loss, seed 0 / seed 2', title_fontsize=10)
leg._legend_box.align = 'left'
fig.text(0.02, 0.015, 'Research runs with every other setting as in this PR. The first three schedules average 0.65 of '
         'the peak; linear to 0 averages 0.5.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('The shape of the decay matters, not its average', x=0.02, ha='left', fontsize=13.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / 'fig4_schedule.png', dpi=200)
print('ok', {k: round(v, 4) for k, v in zip(('meas', 'init', 'speed'), gain.values())},
      [m if m is None else round(m, 1) for *_, m in steps])
