"""Figures for the 30M track 1 PR (cross-token credit read out along each token's real path), one message each.

    /tmp/claude-chunyuan/plotenv/bin/python make_figs.py                  # preview: the research run of the config
    /tmp/claude-chunyuan/plotenv/bin/python make_figs.py --log /home/chunyuan/noise30_pr/train/val/log   # official
--val / --plain / --minutes override the numbers of the log. Train times include the end merge, as record #2's log does
(the research script leaves the merge out of train_seconds, so it is added here).
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LOGS = Path('/home/chunyuan/noise30/logs')
RECORD = ('/home/chunyuan/noise30', '3343973b6f1af119effcdea2622f1fe672c51b1e')  # upstream main, record #2's log
RESEARCH = LOGS / 'd7_30m_e3_s2'  # this PR's config, research run
OUT = Path(__file__).resolve().parent
BLUE, LIGHT, GREY, INK, MUTED = '#1f5fa8', '#a9c1e0', '#8c8c8c', '#222222', '#666666'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b0b0b0', 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'legend.frameon': False})

parser = argparse.ArgumentParser()
parser.add_argument('--val', type=float, help='official val loss (after the end merge)')
parser.add_argument('--plain', type=float, help='official val loss before the end merge')
parser.add_argument('--minutes', type=float, help='official train time, merge included')
parser.add_argument('--log', type=Path, help='the official log folder (config.json, metrics.jsonl, result.json)')
args = parser.parse_args()


def read(folder):
    """A log folder's result.json and its per-update rows; folder 'record' is record #2's log in the upstream repo."""
    if folder == 'record':
        text = lambda name: subprocess.run(['git', '-C', RECORD[0], 'show', f'{RECORD[1]}:train/val/log/{name}'],
                                           check=True, capture_output=True, text=True).stdout
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
    """The run had the node to itself: its Cut updates took at most 5% more than their fastest-decile pace. (Most
    research runs shared GPUs with analysis jobs; their logged times say nothing about the method.)"""
    t = [b['train_seconds'] - a['train_seconds'] for a, b in zip(rows, rows[1:]) if b['phase'] == 'cut']
    return sum(t) <= 1.05 * len(t) * sorted(t)[len(t) // 10]


def num(x, fmt):
    return format(x, fmt).replace('-', '\u2212')


def run(name):
    result, rows = read(LOGS / name)
    return result['val_loss'], minutes(result, rows) if clean(rows) else None


record, record_rows = read('record')
pr, pr_rows = read(args.log or RESEARCH)
official = args.log is not None or args.val is not None
final_val = args.val or pr['val_loss']
final_plain = args.plain or pr['val_loss_plain']
final_min = args.minutes or minutes(pr, pr_rows)
source = 'official log' if official else 'research run'

# 1. each change on top of the previous ones (30M, seed 0); dark: the new cross-token credit, light: recipe and speed
steps = [('Record #2 (its log)', record['val_loss'], minutes(record, record_rows)),
         ('+ faster kernels, Boost head gradient averaged', *run('d4_30m_headavg_base')),
         ('+ Boost for a fixed 244 updates', *run('d4_30m_boost244')),
         ('+ one-sided future probes, moment tables', *run('d4_30m_fast')),
         ('+ ANVIL weight decay 0.75', *run('d5_30m_length')),
         ('+ readout along the real path', *run('d7_30m_readout')),
         ('+ key and gate experiments', *run('d7_30m_readout_kv')),
         ('+ blocks 4-6, input roots', *run('d7_30m_kv_e3')),
         ('+ speed work (this PR)', final_val, final_min)]
steps = [(label, round(v, 4), m) for label, v, m in steps]  # deltas of the shown values, so the column adds up
NEW = range(5, 8)
gain_new = steps[7][1] - steps[4][1]
fig, ax = plt.subplots(figsize=(8, 4.5))
lo, hi = 3.30, 3.445
y = list(range(len(steps)))[::-1]
for k, (yi, (label, v, m)) in enumerate(zip(y, steps)):
    ax.barh(yi, v - lo, left=lo, height=0.62, color=GREY if k == 0 else BLUE if k in NEW else LIGHT)
    txt = f'{v:.4f}' if k == 0 else f'{v:.4f}  ({num(v - steps[k - 1][1], "+.4f")}' + (f', {source})' if yi == 0 else ')')
    ax.text(v + 0.0012, yi, txt, va='center', fontsize=10, color=INK,
            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    if m is not None:
        ax.text(1.01, yi, f'{m:.0f} min', transform=ax.get_yaxis_transform(), va='center', fontsize=10, color=MUTED)
ax.text(1.01, 1.01, 'train\ntime', transform=ax.transAxes, va='bottom', fontsize=9, color=MUTED, linespacing=1.1)
for x, label in [(3.419, 'backprop + Adam\n3.419'), (3.340, 'backprop + ANVIL II\n3.340')]:
    ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0.5)
    ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9, color=MUTED,
            linespacing=1.1)
ax.set_yticks(y, [s for s, *_ in steps], fontsize=10.5)
ax.tick_params(axis='y', length=0)
ax.set_xticks([3.30, 3.35, 3.40], ['3.30', '3.35', '3.40'])
ax.set_xlim(lo, hi)
ax.set_ylim(-0.55, len(steps) - 0.45)
ax.set_xlabel('validation loss (30M, seed 0)')
handles = [Patch(color=BLUE, label=f'new cross-token credit: {num(gain_new, "+.4f")}'),
           Patch(color=LIGHT, label=f'recipe and speed: {num(steps[-1][1] - steps[0][1] - gain_new, "+.4f")}')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.012, 0.9), ncol=2, fontsize=10, handlelength=1.2,
           columnspacing=1.6)
fig.text(0.02, 0.02, 'Train time includes the end merge;\nshown for the runs that had the node to themselves.',
         fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Each change, on top of the previous ones', x=0.02, y=0.975, ha='left', fontsize=13.5)
fig.subplots_adjust(left=0.478, right=0.892, top=0.755, bottom=0.15)
fig.savefig(OUT / 'fig1_steps.png', dpi=200)


# 2. validation loss along training time (validation every 406 updates, untimed; the last two points of each line are
# the final weights before and after the end merge)
def curve(evals, times, plain, merged, end):
    at = {r['step']: r['train_seconds'] / 60 for r in times}
    return [(at[r['step']], r['val_loss']) for r in evals if 'val_loss' in r] + [(times[-1]['train_seconds'] / 60, plain),
                                                                                (end, merged)]


_, rerun = read(LOGS / 'val30_base')  # record #2's script rerun on our node (seed 0): its log has no validation points
pr_evals = pr_rows if any('val_loss' in r for r in pr_rows) else read(RESEARCH)[1]
research_evals = official and (args.log is None or pr_evals is not pr_rows)
lines = [('record #2', GREY, curve(rerun, record_rows, record['val_loss_plain'], record['val_loss'],
                                    minutes(record, record_rows))),
         ('this PR' + ('' if official else ' (research run)'), BLUE,
          curve(pr_evals, pr_rows, final_plain, final_val, final_min))]
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color, pts in lines:
    ax.plot(*zip(*pts), color=color, lw=2, marker='o', ms=4.5, label=label, solid_joinstyle='round')
    ax.annotate(f'{pts[-1][1]:.3f} in {pts[-1][0]:.0f} min', pts[-1], textcoords='offset points', xytext=(9, -1),
                ha='left', va='center', fontsize=11, color=INK)
ax.set_xlim(0, 127)
ax.set_xticks(range(0, 101, 20))
ax.set_ylim(3.25, 4.7)
ax.set_xlabel('training time on 8xH100 (min)')
ax.set_ylabel('validation loss')
ax.grid(axis='y', color='#ececec', lw=0.8)
ax.legend(loc='upper right', fontsize=11)
note = ('Record #2\'s log has no validation during training: those points are our rerun of its script (seed 0), at '
        'its log\'s times.')
if research_evals:
    note += '\nThis PR\'s points before the end are from the research run of the same config.'
fig.text(0.02, 0.015, note, fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Lower loss in less time', x=0.02, ha='left', fontsize=13.5)
fig.tight_layout(rect=(0, 0.04 + 0.03 * note.count('\n'), 1, 1))
fig.savefig(OUT / 'fig2_curves.png', dpi=200)

# 3. the share of what Cut's chain misses at blocks 0-3 that reaches them (backprop only as a ruler: the true gradient at
# each block's mid stream minus the chain's credit = the gap; share = 1 - |added - gap|^2 / |gap|^2; final weights of the
# run before the readout, 4 of the held-out training sequences; research/readout_check.py --only transport)
check = json.loads((LOGS / 'diag' / 'readout_check.json').read_text())['checks']['transport_30m_0_4']['summary']['self']
bars = [('record: carried along the identity', GREY, [check[f'mid{i}']['filled_identity'] for i in range(4)]),
        ('this PR: read out along the real path', BLUE, [check[f'mid{i}']['filled'] for i in range(4)])]
fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.3
for k, (label, color, vals) in enumerate(bars):
    xs = [i + (k - 0.5) * (w + 0.02) for i in range(4)]
    ax.bar(xs, vals, width=w, color=color, label=label)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f'{v:.2f}'.lstrip('0'), ha='center', va='bottom', fontsize=10.5, color=INK)
ax.set_xticks(range(4), [f'block {i}' for i in range(4)])
ax.tick_params(axis='x', length=0)
ax.set_ylim(0, 1.08)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '.25', '.50', '.75', '1'])
ax.set_ylabel('share recovered')
ax.grid(axis='y', color='#ececec', lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=10.5)
fig.text(0.02, 0.015, 'Backprop only as a ruler. Gap: the true gradient at each block\'s mid stream minus Cut\'s chain '
         'credit (the cross-token part).\nShare recovered: 1 \u2212 |added credit \u2212 gap|\u00b2 / |gap|\u00b2. '
         'Final weights of the run before the readout, 4 held-out sequences.', fontsize=8.5, color=MUTED, va='bottom')
fig.suptitle('Most of the cross-token credit now reaches blocks 0-3', x=0.02, ha='left', fontsize=13.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / 'fig3_credit.png', dpi=200)
print(f'ok: {source}, final {final_val:.4f} / {final_plain:.4f} in {final_min:.1f} min')
