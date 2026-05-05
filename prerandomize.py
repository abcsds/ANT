#!/usr/bin/env python3
"""Pre-randomize trial sequences for the ANT experiment.

Generates K block lists under lists/ (one CSV per block list, each a randomly
permuted concatenation of the 24 base conditions repeated REPS_PER_BLOCK
times) and 1000 per-participant schedules under schedules/ that pick block
lists in a permuted order. Each participant's schedule has MAX_BLOCKS rows;
the experiment skips rows where block > nBlocks.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import html
import random
import re
from collections import Counter
from pathlib import Path

# --- Experiment parameters (mirror the PsychoPy task) -----------------------

# Trial timing — must match ANT.psyexp. Fix is jittered uniformly:
#   fix_duration = FIX_BASE + uniform(0, FIX_JITTER)
# so its mean is FIX_BASE + FIX_JITTER / 2.
FIX_BASE = 0.4
FIX_JITTER = 1.2
CUE_DURATION = 0.4         # cue routine total length
CUE_STIM_DURATION = 0.1    # cue text on-screen time within the cue routine
TARGET_MAX_DURATION = 1.7  # target routine timeout (response ends it earlier)

# Mean response time used only to *estimate* duration in the report.
# Healthy adult ANT mean RT is ~550 ms; pad a little for occasional misses
# that run to TARGET_MAX_DURATION.
ASSUMED_MEAN_RT = 0.6

# Trial structure: 4 cue types × 6 target stimuli = 24 base conditions;
# each block contains REPS_PER_BLOCK copies of each condition, randomly
# permuted. With the canonical 24 / 5 these are 120-trial blocks.
CUE_TYPES = ["NC", "CC", "DC", "SC"]
TARGETS = [
    ("<<<<<", "f", "left",  "congruent"),
    (">>>>>", "j", "right", "congruent"),
    (">><>>", "f", "left",  "incongruent"),
    ("<<><<", "j", "right", "incongruent"),
    ("--<--", "f", "left",  "neutral"),
    ("-->--", "j", "right", "neutral"),
]
REPS_PER_BLOCK = 5
TRIALS_PER_BLOCK = len(CUE_TYPES) * len(TARGETS) * REPS_PER_BLOCK  # 120

INSTRUCTIONS_DURATION = 60   # seconds reserved for welcome + intro
INTER_BLOCK_BREAK = 60       # 1-min self-paced break between blocks

N_BLOCK_LISTS = 10           # block_a..block_j
LIST_LETTERS = [chr(ord("a") + i) for i in range(N_BLOCK_LISTS)]
MAX_BLOCKS = N_BLOCK_LISTS   # so each participant uses each list at most once

N_PARTICIPANTS = 1000        # 3-digit codes 000..999

DEFAULT_SEED = 20260505      # change this to regenerate; LOG.md records each new seed


# --- Helpers ----------------------------------------------------------------


def base_conditions() -> list[tuple[str, str, str]]:
    """The 24 base trials = full crossing of cue × target."""
    rows = []
    for cue in CUE_TYPES:
        for target, correct, _, _ in TARGETS:
            rows.append((cue, target, correct))
    assert len(rows) == 24
    return rows


def generate_block_list(rng: random.Random) -> list[tuple[str, str, str]]:
    """Random permutation of REPS_PER_BLOCK copies of every base condition."""
    trials = base_conditions() * REPS_PER_BLOCK
    rng.shuffle(trials)
    return trials


def verify_block_list(trials: list[tuple[str, str, str]]) -> None:
    assert len(trials) == TRIALS_PER_BLOCK, (
        f"block has {len(trials)} trials, expected {TRIALS_PER_BLOCK}"
    )
    counts = Counter(trials)
    expected = {row: REPS_PER_BLOCK for row in base_conditions()}
    assert counts == expected, (
        f"block condition counts unbalanced: {dict(counts)} vs expected {expected}"
    )
    # Sanity: per-cue and per-correct totals.
    cue_counts = Counter(t[0] for t in trials)
    expected_per_cue = TRIALS_PER_BLOCK // len(CUE_TYPES)
    for cue in CUE_TYPES:
        assert cue_counts[cue] == expected_per_cue, (
            f"cue {cue}: {cue_counts[cue]} != {expected_per_cue}"
        )
    correct_counts = Counter(t[2] for t in trials)
    assert correct_counts["f"] == correct_counts["j"] == TRIALS_PER_BLOCK // 2, (
        f"correct counts unbalanced: {dict(correct_counts)}"
    )


def write_block_csv(path: Path, trials: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cue", "target", "correct"])
        w.writerows(trials)


def generate_schedule(rng: random.Random) -> list[tuple[int, str, str]]:
    """One participant's schedule: MAX_BLOCKS rows, list-letters permuted."""
    perm = rng.sample(LIST_LETTERS, len(LIST_LETTERS))
    return [
        (block, perm[block - 1], f"lists/block_{perm[block - 1]}.csv")
        for block in range(1, MAX_BLOCKS + 1)
    ]


def verify_schedule(rows: list[tuple[int, str, str]]) -> None:
    assert len(rows) == MAX_BLOCKS, f"schedule has {len(rows)} rows, expected {MAX_BLOCKS}"
    expected_blocks = list(range(1, MAX_BLOCKS + 1))
    assert [r[0] for r in rows] == expected_blocks, (
        "schedule rows must be ordered by block ascending"
    )
    letters = [r[1] for r in rows]
    assert sorted(letters) == LIST_LETTERS, (
        f"schedule must use each list letter exactly once: got {sorted(letters)}"
    )
    for block, letter, conds in rows:
        assert conds == f"lists/block_{letter}.csv", (
            f"row {block}: condsFile {conds!r} does not match letter {letter!r}"
        )


def write_schedule(path: Path, rows: list[tuple[int, str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "list_letter", "condsFile"])
        w.writerows(rows)


def read_last_seed(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"seed:\s*(-?\d+)")
    last = None
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            last = int(m.group(1))
    return last


def append_log(log_path: Path, seed: int) -> None:
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    entry = f"- {timestamp} — seed: {seed}\n"
    if not log_path.exists():
        header = (
            "# Pre-randomization seed log\n\n"
            "Each entry records a run of `prerandomize.py` whose seed differed\n"
            "from the previous run. This is the audit trail for reproducibility:\n"
            "all CSVs in `lists/` and `schedules/` were generated by the most\n"
            "recent seed below.\n\n"
        )
        log_path.write_text(header + entry)
    else:
        with log_path.open("a") as f:
            f.write(entry)


# --- Duration estimation ---------------------------------------------------


def mean_trial_duration() -> float:
    """Mean wall-clock seconds per trial (used for duration estimates only)."""
    mean_fix = FIX_BASE + FIX_JITTER / 2
    return mean_fix + CUE_DURATION + ASSUMED_MEAN_RT


def block_duration(n_trials: int = TRIALS_PER_BLOCK) -> float:
    return n_trials * mean_trial_duration()


def task_time(n_blocks: int) -> float:
    """Seconds for the task (without instructions): blocks + inter-block breaks."""
    breaks = max(n_blocks - 1, 0)
    return n_blocks * block_duration() + breaks * INTER_BLOCK_BREAK


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s" if sec else f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s" if sec else f"{m}m"


# --- Report -----------------------------------------------------------------


CUE_COLORS = {
    "NC": "#bdc3c7",  # gray
    "CC": "#3498db",  # blue
    "DC": "#9b59b6",  # purple
    "SC": "#e67e22",  # orange
}
CONGRUENCY_COLORS = {
    "congruent":   "#27ae60",
    "incongruent": "#c0392b",
    "neutral":     "#7f8c8d",
}
TARGET_INFO = {target: (correct, direction, congruency)
               for target, correct, direction, congruency in TARGETS}


def _svg_bars(items, max_val=None, label_w: int = 130, bar_max: int = 320,
              row_h: int = 22, color: str = "#3498db") -> str:
    if not items:
        return ""
    if max_val is None:
        max_val = max(v for _, v in items) or 1
    width = label_w + bar_max + 60
    height = row_h * len(items) + 4
    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    for i, (label, value) in enumerate(items):
        y = i * row_h
        bar_w = (value / max_val) * bar_max
        parts.append(
            f'<text x="{label_w - 6}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12" fill="#2c3e50">'
            f'{html.escape(str(label))}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{bar_w:.1f}" height="{row_h - 6}" '
            f'fill="{color}" rx="2"/>'
            f'<text x="{label_w + bar_w + 5:.1f}" y="{y + row_h / 2 + 4:.1f}" '
            f'font-family="sans-serif" font-size="12" fill="#2c3e50">{value}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_timeline(trials: list[tuple[str, str, str]],
                  cell_w: int = 18, cell_h: int = 26, gap: int = 1) -> str:
    """One row per list. Cell colour = cue type; arrow inside encodes direction."""
    n = len(trials)
    width = n * (cell_w + gap)
    height = cell_h + 18
    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    for i, (cue, target, _correct) in enumerate(trials):
        x = i * (cell_w + gap)
        cue_color = CUE_COLORS[cue]
        _, direction, congruency = TARGET_INFO[target]
        arrow = "←" if direction == "left" else "→"
        # Sub-bar at top encodes congruency; main fill encodes cue.
        parts.append(
            f'<rect x="{x}" y="0" width="{cell_w}" height="{cell_h}" '
            f'fill="{cue_color}" rx="2"/>'
            f'<rect x="{x}" y="0" width="{cell_w}" height="3" '
            f'fill="{CONGRUENCY_COLORS[congruency]}"/>'
            f'<text x="{x + cell_w / 2:.1f}" y="{cell_h / 2 + 6:.1f}" '
            f'text-anchor="middle" fill="#ffffff" '
            f'font-family="sans-serif" font-size="13" font-weight="700">{arrow}</text>'
        )
        if i % 10 == 0 or i == n - 1:
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{cell_h + 14}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'fill="#7f8c8d">{i}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _schedule_table_html() -> str:
    head = (
        "<thead>"
        "<tr>"
        "<th>nBlocks (K)</th>"
        "<th>Instructions</th>"
        "<th>Task (blocks + breaks)</th>"
        "<th>Total</th>"
        "</tr></thead>"
    )
    instr = _format_duration(INSTRUCTIONS_DURATION)
    rows = []
    for k in range(1, MAX_BLOCKS + 1):
        task = task_time(k)
        total = INSTRUCTIONS_DURATION + task
        rows.append(
            f"<tr><td>{k}</td><td>{instr}</td>"
            f"<td>{_format_duration(task)}</td>"
            f"<td>{_format_duration(total)}</td></tr>"
        )
    return f'<table class="schedule">{head}<tbody>{"".join(rows)}</tbody></table>'


def _timing_table_html() -> str:
    rows = [
        ("Fixation (D1)",
         f"{FIX_BASE:g} + uniform(0, {FIX_JITTER:g}) s "
         f"&rarr; mean {FIX_BASE + FIX_JITTER / 2:g} s"),
        ("Cue routine total",   f"{CUE_DURATION:g} s"),
        ("Cue stim on-screen",  f"{CUE_STIM_DURATION:g} s"),
        ("Target max",          f"{TARGET_MAX_DURATION:g} s (or until F/J keypress)"),
        ("Assumed mean RT",     f"{ASSUMED_MEAN_RT:g} s (estimation only)"),
        ("Trials per block",    f"{TRIALS_PER_BLOCK} "
                                f"({len(CUE_TYPES)} cues &times; {len(TARGETS)} targets "
                                f"&times; {REPS_PER_BLOCK} reps)"),
        ("Mean trial duration", f"{mean_trial_duration():.2f} s"),
        ("Mean block duration", f"{_format_duration(block_duration())}"),
        ("Inter-block break",   f"{INTER_BLOCK_BREAK} s"),
        ("Block lists",         f"{N_BLOCK_LISTS} (block_a.csv &hellip; "
                                f"block_{LIST_LETTERS[-1]}.csv)"),
        ("Max blocks per run",  f"{MAX_BLOCKS}"),
        ("Participant schedules", f"{N_PARTICIPANTS} (000.csv &hellip; "
                                  f"{N_PARTICIPANTS - 1:03d}.csv)"),
    ]
    body = "".join(
        f'<tr><th scope="row">{label}</th><td>{value}</td></tr>' for label, value in rows
    )
    return f'<table class="timings"><tbody>{body}</tbody></table>'


def _sample_schedule_html(rows: list[tuple[int, str, str]]) -> str:
    body = "".join(
        f"<tr><td>{block}</td><td>{letter}</td><td>{conds}</td></tr>"
        for block, letter, conds in rows
    )
    head = "<thead><tr><th>block</th><th>list_letter</th><th>condsFile</th></tr></thead>"
    return f'<table class="schedule">{head}<tbody>{body}</tbody></table>'


def _legend_html() -> str:
    cue_swatches = "".join(
        f'<span class="swatch" style="background:{CUE_COLORS[c]}"></span>{c}&nbsp;&nbsp;'
        for c in CUE_TYPES
    )
    cong_swatches = "".join(
        f'<span class="swatch tall" style="background:{CONGRUENCY_COLORS[c]}"></span>{c}&nbsp;&nbsp;'
        for c in ("congruent", "incongruent", "neutral")
    )
    return (
        '<div class="legend">'
        '<strong>cell fill (cue):</strong>&nbsp;' + cue_swatches +
        '&nbsp;&nbsp;<strong>top stripe (flanker):</strong>&nbsp;' + cong_swatches +
        '&nbsp;&nbsp;<strong>arrow:</strong>&nbsp;&larr; left&nbsp;&nbsp;&rarr; right'
        '</div>'
    )


def render_html_report(report_path: Path,
                       lists: list[tuple[str, list[tuple[str, str, str]]]],
                       sample_schedule: list[tuple[int, str, str]],
                       seed: int) -> None:
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1400px; margin: 1em auto; padding: 0 1em; color: #2c3e50; }
    h1 { margin-bottom: 0.2em; }
    h2 { margin-top: 1.5em; }
    h3 { margin-top: 1.4em; margin-bottom: 0.4em; }
    .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 1em; }
    .legend { margin: 0.5em 0 1em; font-size: 0.9em; color: #34495e; }
    .legend .swatch { display: inline-block; width: 14px; height: 14px;
                      border-radius: 3px; vertical-align: middle; margin-right: 4px; }
    .legend .swatch.tall { height: 6px; }
    .summary { display: flex; gap: 2em; flex-wrap: wrap; }
    .summary > div { flex: 1; min-width: 320px; }
    .list-block { margin: 0.6em 0; padding: 0.6em 1em;
                  border: 1px solid #ecf0f1; border-radius: 6px; background: #fafbfc;
                  overflow-x: auto; }
    .list-header { display: flex; justify-content: space-between;
                   align-items: baseline; margin-bottom: 0.3em; gap: 1em;
                   flex-wrap: wrap; }
    .list-name { font-weight: 600;
                 font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .list-stats { color: #7f8c8d; font-size: 0.85em; }
    .schedule, .timings { border-collapse: collapse; margin: 0.5em 0 1em;
                          font-family: -apple-system, sans-serif; font-size: 13px; }
    .schedule th, .schedule td,
    .timings th, .timings td { padding: 6px 14px;
                                border-bottom: 1px solid #ecf0f1; text-align: right; }
    .schedule th:first-child, .schedule td:first-child,
    .timings th { text-align: left; }
    .schedule th, .timings th { background: #ecf0f1; font-weight: 600; }
    .schedule tbody tr:hover, .timings tbody tr:hover { background: #f4f6f7; }
    .schedule-note { color: #7f8c8d; font-size: 0.85em; margin: 0 0 1.5em; }
    """

    # Aggregate counts across all lists.
    cue_counter: Counter = Counter()
    target_counter: Counter = Counter()
    cong_counter: Counter = Counter()
    for _, trials in lists:
        for cue, target, _ in trials:
            cue_counter[cue] += 1
            target_counter[target] += 1
            cong_counter[TARGET_INFO[target][2]] += 1

    cue_bar = _svg_bars(
        [(c, cue_counter[c]) for c in CUE_TYPES],
        color="#3498db", label_w=60, bar_max=320,
    )
    cong_bar = _svg_bars(
        [(c, cong_counter[c]) for c in ("congruent", "incongruent", "neutral")],
        color="#27ae60", label_w=110, bar_max=320,
    )
    target_bar = _svg_bars(
        [(t, target_counter[t]) for t, *_ in TARGETS],
        color="#9b59b6", label_w=80, bar_max=320,
    )

    per_list_html = []
    for name, trials in lists:
        cues_in_list = Counter(t[0] for t in trials)
        cue_summary = " ".join(f"{c}:{cues_in_list[c]}" for c in CUE_TYPES)
        per_list_html.append(
            f'<div class="list-block">'
            f'<div class="list-header">'
            f'<span class="list-name">{html.escape(name)}.csv</span>'
            f'<span class="list-stats">{len(trials)} trials &middot; {cue_summary}</span>'
            f"</div>{_svg_timeline(trials)}</div>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANT lists report</title>
<style>{css}</style>
</head>
<body>
<h1>ANT lists report</h1>
<p class="meta">Generated {timestamp} &middot; seed: {seed} &middot;
{len(lists)} block lists ({TRIALS_PER_BLOCK} trials each) &middot;
{N_PARTICIPANTS} participant schedules</p>

<h2>Timing parameters</h2>
{_timing_table_html()}

<h2>Experiment schedule</h2>
<p class="schedule-note">Wall-clock estimate as a function of <code>nBlocks</code>
(experimenter-set, 1&hellip;{MAX_BLOCKS}). The task time covers
<em>K</em> blocks of {TRIALS_PER_BLOCK} trials, each trial averaging
{mean_trial_duration():.2f} s (mean fixation {FIX_BASE + FIX_JITTER / 2:g} s
+ cue {CUE_DURATION:g} s + assumed mean RT {ASSUMED_MEAN_RT:g} s), plus a
{INTER_BLOCK_BREAK} s break between every consecutive pair of blocks. Add the
&ldquo;Instructions&rdquo; column ({_format_duration(INSTRUCTIONS_DURATION)},
self-paced welcome + intro) to get the wall-clock total.</p>
{_schedule_table_html()}

<h2>Sample participant schedule (000.csv)</h2>
<p class="schedule-note">Each of the {N_PARTICIPANTS} participant codes maps to
a permutation of the {N_BLOCK_LISTS} block lists. The experiment reads the
participant's schedule and stops once <code>block &gt; nBlocks</code>.</p>
{_sample_schedule_html(sample_schedule)}

<h2>Aggregate counts across all {len(lists)} lists</h2>
{_legend_html()}
<div class="summary">
<div><h3>Cue type</h3>{cue_bar}</div>
<div><h3>Flanker congruency</h3>{cong_bar}</div>
<div><h3>Target arrows</h3>{target_bar}</div>
</div>

<h2>Per-list timelines</h2>
{_legend_html()}
{''.join(per_list_html)}

</body>
</html>
"""
    report_path.write_text(doc)


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--output", type=Path, default=Path("lists"),
                        help="output directory for block lists (default: lists)")
    parser.add_argument("--schedules", type=Path, default=Path("schedules"),
                        help="output directory for participant schedules (default: schedules)")
    parser.add_argument("--log", type=Path, default=Path("LOG.md"),
                        help="seed log file (default: LOG.md)")
    parser.add_argument("--report", type=Path, default=Path("docs/report.html"),
                        help="HTML report file (default: docs/report.html)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.schedules.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    master_rng = random.Random(args.seed)

    # 1. Block lists
    lists: list[tuple[str, list[tuple[str, str, str]]]] = []
    for letter in LIST_LETTERS:
        sub_rng = random.Random(master_rng.getrandbits(64))
        trials = generate_block_list(sub_rng)
        verify_block_list(trials)
        name = f"block_{letter}"
        write_block_csv(args.output / f"{name}.csv", trials)
        lists.append((name, trials))

    # 2. Per-participant schedules
    sample_schedule: list[tuple[int, str, str]] | None = None
    for pid in range(N_PARTICIPANTS):
        sub_rng = random.Random(master_rng.getrandbits(64))
        rows = generate_schedule(sub_rng)
        verify_schedule(rows)
        write_schedule(args.schedules / f"{pid:03d}.csv", rows)
        if pid == 0:
            sample_schedule = rows
    assert sample_schedule is not None

    # 3. HTML report
    render_html_report(args.report, lists, sample_schedule, args.seed)

    # 4. Seed log
    last_seed = read_last_seed(args.log)
    if last_seed != args.seed:
        append_log(args.log, args.seed)
        log_msg = f"seed changed ({last_seed} -> {args.seed}); appended to {args.log}"
    else:
        log_msg = f"seed unchanged ({args.seed}); {args.log} not updated"

    print(
        f"Wrote {len(lists)} block lists ({TRIALS_PER_BLOCK} trials each) to {args.output}/"
    )
    print(f"Wrote {N_PARTICIPANTS} participant schedules to {args.schedules}/")
    print(f"Wrote report to {args.report}")
    print(log_msg)


if __name__ == "__main__":
    main()
