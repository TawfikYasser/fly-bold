import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

experiment_files = {
    'EXP_103100001':   '/Users/tawfik/DeFeC3/flybold/103100001.txt',
    'EXP_103101004':   '/Users/tawfik/DeFeC3/flybold/103101004.txt',
    'EXP_3010100002':  '/Users/tawfik/DeFeC3/flybold/3010100002.txt',
    'EXP_3010101001':  '/Users/tawfik/DeFeC3/flybold/3010101001.txt',
}

# Pairs to plot together (share the same subplot)
PAIRS = [
    ('EXP_3010100002', 'EXP_3010101001'),
    ('EXP_103100001',  'EXP_103101004'),
]

PAIR_COLORS = {
    'EXP_103100001':  '#1f77b4',
    'EXP_103101004':  '#ff7f0e',
    'EXP_3010100002': '#2ca02c',
    'EXP_3010101001': '#d62728',
}


def extract_all_trials(filepath):
    """
    Returns:
      trials   : list of (trial_num, [mAP@0.5 per round])  — includes pruned (short) trials
      best_idx : index into trials of the best trial (or None)
      best_mAP : actual best mAP value from OPTUNA or None
    """
    with open(filepath, 'r') as f:
        content = f.read()

    parts = re.split(r'STARTING FL TRIAL\s+\[optuna_trial_(\d+)\]', content)
    # parts = [preamble, trial_num, body, trial_num, body, ...]

    trials = []
    for i in range(1, len(parts) - 1, 2):
        trial_num = int(parts[i])
        body      = parts[i + 1]
        maps = [float(v) for v in re.findall(r'\[SERVER\] Validation mAP@0\.5:\s+([\d.]+)', body)]
        trials.append((trial_num, maps))   # keep even if maps is empty

    best_match = re.search(r'OPTUNA COMPLETE -- Best trial #(\d+)', content)
    best_trial_num = int(best_match.group(1)) if best_match else None

    # Extract actual best mAP from OPTUNA message
    best_mAP_match = re.search(r'OPTUNA COMPLETE -- Best trial #\d+\s+mAP@0\.5\s+=\s+([\d.]+)', content)
    best_mAP = float(best_mAP_match.group(1)) if best_mAP_match else None

    best_idx = None
    if best_trial_num is not None:
        for idx, (tnum, _) in enumerate(trials):
            if tnum == best_trial_num:
                best_idx = idx
                break

    # Best trial was pruned before execution (not in log) — fall back to highest peak mAP
    if best_idx is None and trials:
        best_idx = max(range(len(trials)), key=lambda i: max(trials[i][1]) if trials[i][1] else -1)
        fallback_peak = max(trials[best_idx][1]) if trials[best_idx][1] else 0
        print(f"  ⚠️  Best trial #{best_trial_num} pruned — falling back to trial "
              f"#{trials[best_idx][0]} (peak mAP={fallback_peak:.4f}, actual best={best_mAP})")

    return trials, best_idx, best_mAP


# ── Collect all data ──────────────────────────────────────────────────────────
all_data = {}
for name, path in experiment_files.items():
    trials, best_idx, best_mAP = extract_all_trials(path)
    all_data[name] = (trials, best_idx, best_mAP)
    counts = [len(m) for _, m in trials]
    max_r  = max(counts) if counts else 0
    best_trial_num = trials[best_idx][0] if best_idx is not None else '?'
    print(f"{name}: {len(trials)} trials | max_rounds={max_r} | "
          f"pruned={sum(1 for c in counts if c < max_r)} | "
          f"best_idx={best_idx} (trial #{best_trial_num}) | best_mAP={best_mAP}")


def plot_pair(ax, exp_a, exp_b):
    """
    Plot two experiments on the same axes, aligned by trial position.
    Each trial occupies `max_rounds` slots on the x-axis (fixed width),
    so pruned trials simply terminate early within their slot.
    """
    trials_a, best_a, best_mAP_a = all_data[exp_a]
    trials_b, best_b, best_mAP_b = all_data[exp_b]

    assert len(trials_a) == len(trials_b), "Paired experiments must have the same trial count"
    n_trials = len(trials_a)

    # Fixed slot width = max rounds across both experiments
    max_rounds = max(
        max((len(m) for _, m in trials_a), default=0),
        max((len(m) for _, m in trials_b), default=0),
    )

    slot_width = max_rounds   # each trial occupies this many x-units

    def build_xy(trials, best_idx):
        """Build x/y arrays and boundary positions using fixed-width trial slots."""
        xs, ys = [], []
        boundaries = []
        best_xs, best_ys = [], []

        for pos, (tnum, maps) in enumerate(trials):
            slot_start = pos * slot_width
            boundaries.append(slot_start)
            for r, mAP in enumerate(maps):
                x = slot_start + r
                xs.append(x)
                ys.append(mAP)
                if pos == best_idx:
                    best_xs.append(x)
                    best_ys.append(mAP)

        return (np.array(xs), np.array(ys),
                np.array(best_xs), np.array(best_ys),
                boundaries)

    xs_a, ys_a, bx_a, by_a, bounds_a = build_xy(trials_a, best_a)
    xs_b, ys_b, bx_b, by_b, bounds_b = build_xy(trials_b, best_b)

    total_width = n_trials * slot_width

    # ── Trial boundary lines & labels ──────────────────────────────────────
    for pos in range(n_trials):
        x_sep = pos * slot_width - 0.5
        ax.axvline(x=x_sep, color='gray', linewidth=0.5, linestyle='--', alpha=0.35)

        n_rounds_a = len(trials_a[pos][1])
        n_rounds_b = len(trials_b[pos][1])
        is_pruned = (n_rounds_a < max_rounds) or (n_rounds_b < max_rounds)

        # ax.text(pos * slot_width + slot_width / 2 - 0.5,
        #         1.04, f'T{pos+1}',
        #         fontsize=7, ha='center', va='bottom',
        #         transform=ax.get_xaxis_transform(),
        #         color='#c00000' if is_pruned else 'gray',
        #         fontweight='bold' if is_pruned else 'normal')

    # ── Plot exp_a ────────────────────────────────────────────────────────
    color_a = PAIR_COLORS[exp_a]
    ax.plot(xs_a, ys_a, color=color_a, linewidth=1.4, alpha=0.8, zorder=3)
    ax.scatter(xs_a, ys_a, color=color_a, s=20, zorder=4, alpha=0.9)
    if len(bx_a):
        ax.plot(bx_a, by_a, color=color_a, linewidth=4, alpha=0.35, zorder=2)
        ax.scatter(bx_a, by_a, color='gold', s=55, zorder=5,
                   edgecolors=color_a, linewidths=1.2)

    # ── Plot exp_b ────────────────────────────────────────────────────────
    color_b = PAIR_COLORS[exp_b]
    ax.plot(xs_b, ys_b, color=color_b, linewidth=1.4, alpha=0.8,
            linestyle='--', zorder=3)
    ax.scatter(xs_b, ys_b, color=color_b, s=20, zorder=4, alpha=0.9, marker='s')
    if len(bx_b):
        ax.plot(bx_b, by_b, color=color_b, linewidth=4, alpha=0.35,
                linestyle='--', zorder=2)
        ax.scatter(bx_b, by_b, color='gold', s=55, zorder=5,
                   edgecolors=color_b, linewidths=1.2, marker='s')

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xlim(-0.8, total_width - 0.2)
    ax.set_ylabel('Val mAP@0.5', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

    step = 5 if n_trials > 15 else 2
    tick_positions = [pos * slot_width + slot_width / 2 - 0.5 for pos in range(0, n_trials, step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pos + 1) for pos in range(0, n_trials, step)], fontsize=9)
    ax.set_xlabel('Trials', fontsize=10)

    # Legend
    # def best_label(exp, trials, best_idx, best_ys, best_mAP):
    #     tnum = trials[best_idx][0] if best_idx is not None else '?'
    #     peak = f'{best_mAP:.4f}' if best_mAP else (f'{max(best_ys):.4f}' if len(best_ys) else 'n/a')
    #     return f'{exp}  (best trial #{tnum}, peak mAP={peak})'

    # handles = [
    #     mpatches.Patch(color=color_a, label=best_label(exp_a, trials_a, best_a, by_a, best_mAP_a)),
    #     mpatches.Patch(color=color_b, label=best_label(exp_b, trials_b, best_b, by_b, best_mAP_b)),
    #     mpatches.Patch(color='gold',  label='Best trial (gold dots)'),
    # ]
    # ax.legend(handles=handles, fontsize=9, loc='lower right', framealpha=0.9,
    #           frameon=True, fancybox=True)

    n_pruned_a = sum(1 for _, m in trials_a if len(m) < max_rounds)
    n_pruned_b = sum(1 for _, m in trials_b if len(m) < max_rounds)
    # ax.set_title(
    #     f'{exp_a}  vs  {exp_b}\n'
    #     f'{n_trials} trials × up to {max_rounds} rounds  |  '
    #     f'pruned: {exp_a}={n_pruned_a}, {exp_b}={n_pruned_b}',
    #     fontsize=11, fontweight='bold', pad=15
    # )


# ── Build one figure per pair ─────────────────────────────────────────────────
pair_filenames = [
    'all_trials_map_plot_30trials_BOHB.png',
    'all_trials_map_plot_10trials_HO.png',
]

for (exp_a, exp_b), fname in zip(PAIRS, pair_filenames):
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    plot_pair(ax, exp_a, exp_b)
    # fig.suptitle('Flybold — All Trials & Rounds: Validation mAP@0.5',
    #              fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = f'/Users/tawfik/DeFeC3/flybold/{fname}'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved → {output_path}")
    plt.show()