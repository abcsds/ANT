#!/usr/bin/env python3
"""Build per-participant HTML reports from ANT PsychoPy CSVs.

Scans ``data/`` for finished runs, writes ``docs/reports/<slug>.html`` for each
and rebuilds ``docs/index.html`` so it lists every available report.

Each report has four tabs:
  * **Overview** — run metadata, per-block A/O/C scores table, aggregate
    scores card, congruency-by-cue mean-RT card.
  * **Per-block** — one card per block with the trial timeline (Bokeh)
    and a per-cue RT histogram.
  * **RT distributions** — two Bokeh figures: one coloured by cue type,
    one by flanker congruency. Click legend entries to hide a series.
  * **Bayesian** — analytic Normal-Normal posterior for alerting,
    orienting, and conflict given Fan & Posner literature priors.
    Prior-vs-posterior density overlays, posterior summary, reference
    distributions ghosted underneath.

Visual style mirrors ``docs/report.html`` (the prerandomization page) and
NBack's report pages: system font stack, ink ``#2c3e50``, muted ``#7f8c8d``,
accent ``#3498db``, white cards on ``#fafbfc``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import html
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import INLINE
from scipy import stats

# --- Constants --------------------------------------------------------------

DATA_DIR = Path("data")
REPORTS_DIR = Path("docs/reports")
INDEX_PATH = Path("docs/index.html")

CUE_TYPES = ["NC", "CC", "DC", "SC"]
CUE_LABEL = {
    "NC": "No cue",
    "CC": "Centre cue",
    "DC": "Double cue",
    "SC": "Spatial cue",
}
CUE_COLOURS = {
    "NC": "#94a3b8",  # slate
    "CC": "#3498db",  # blue
    "DC": "#9b59b6",  # purple
    "SC": "#e67e22",  # orange
}

CONGRUENCY = ["congruent", "incongruent", "neutral"]
CONGRUENCY_COLOURS = {
    "congruent":   "#27ae60",  # green
    "incongruent": "#c0392b",  # red
    "neutral":     "#7f8c8d",  # gray
}

# (target stimulus → flanker congruency)
FLANKER_OF = {
    "<<<<<": "congruent",
    ">>>>>": "congruent",
    ">><>>": "incongruent",
    "<<><<": "incongruent",
    "--<--": "neutral",
    "-->--": "neutral",
}

SCORE_LABELS = {
    "alerting":  "Alerting",
    "orienting": "Orienting",
    "conflict":  "Conflict",
}
SCORE_COLOURS = {
    "alerting":  "#1abc9c",  # teal
    "orienting": "#f39c12",  # amber
    "conflict":  "#c0392b",  # red
}
SCORE_FORMULA = {
    "alerting":  "RT(NC) − RT(DC)",
    "orienting": "RT(CC) − RT(SC)",
    "conflict":  "RT(IC) − RT(CG)",
}

# Literature priors (Normal mean / std in ms). Used by the Bayesian tab.
# Centres come from the per-block ANT.jl model used previously; widths are
# the σ=100 ms used in the Julia Turing model's likelihood, which we keep
# for prior on μ as well so the per-block sample mean dominates as n grows.
PRIOR = {
    "alerting":  {"mu": 40.0, "sigma": 100.0},
    "orienting": {"mu": 50.0, "sigma": 100.0},
    "conflict":  {"mu": 98.0, "sigma": 100.0},
}
# Likelihood σ: per-block scores are noisy estimates of the underlying
# subject mean; 100 ms reflects typical between-block variability in ANT.
LIKE_SIGMA = 100.0

# Reference (Fan & Posner-derived) population distributions, rendered as
# ghosted overlays for context.
REF_DIST = {
    "alerting":  {"mu": 47.0, "sigma": 18.0},
    "orienting": {"mu": 51.0, "sigma": 21.0},
    "conflict":  {"mu": 84.0, "sigma": 25.0},
}


# --- CSV loading ------------------------------------------------------------


def _to_float(v):
    if v is None or v == "" or v == "None":
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _format_duration(seconds) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    s = int(round(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s"


def load_run(csv_path: Path) -> dict | None:
    """Read one ANT PsychoPy CSV. Return None for runs with no trials."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # Strip BOM that PsychoPy sometimes prepends to the first column.
    df.columns = [c.lstrip("﻿") for c in df.columns]

    required = ["block", "list_letter", "cue", "target", "correct",
                "key_resp_target.keys", "key_resp_target.rt"]
    if any(c not in df.columns for c in required):
        return None

    trials = df[
        (df["block"] != "")
        & (df["cue"] != "")
        & (df.get("trials.thisN", pd.Series([""] * len(df))) != "")
    ].copy()
    if trials.empty:
        return None

    trials["block"] = trials["block"].astype(int)
    trials["rt"] = trials["key_resp_target.rt"].apply(_to_float)
    trials["pressed"] = trials["key_resp_target.keys"].apply(
        lambda v: v if v not in ("", "None") else ""
    )
    trials["congruency"] = trials["target"].map(FLANKER_OF)
    trials["correct_resp"] = trials["pressed"] == trials["correct"]
    # Within-block trial index (1-based for display).
    trials["trial_idx"] = trials.groupby("block").cumcount() + 1

    # Metadata
    def first(col, default=""):
        if col not in df.columns:
            return default
        for v in df[col]:
            if v not in ("", "None"):
                return v
        return default

    if "thisRow.t" in df.columns:
        ts = pd.to_numeric(df["thisRow.t"], errors="coerce").dropna()
        duration_s = float(ts.max() - ts.min()) if len(ts) >= 2 else None
    else:
        duration_s = None

    blocks = []
    for block_num, block_df in trials.groupby("block", sort=True):
        block_df = block_df.sort_values("trial_idx").reset_index(drop=True)
        blocks.append(_summarise_block(block_num, block_df))

    return {
        "csv_path": csv_path,
        "csv_name": csv_path.name,
        "report_slug": _slugify(csv_path.stem),
        "participant": first("participant") or "anon",
        "session": first("session"),
        "date_str": first("date"),
        "psychopy_version": first("psychopyVersion"),
        "n_blocks_setting": first("nBlocks"),
        "duration_s": duration_s,
        "trials": trials,
        "blocks": blocks,
    }


def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def _summarise_block(block_num: int, block_df: pd.DataFrame) -> dict:
    list_letter = block_df["list_letter"].iloc[0] if "list_letter" in block_df else ""
    n_trials = len(block_df)
    correct = block_df[block_df["correct_resp"]]
    n_correct = len(correct)
    n_errors = n_trials - n_correct
    n_no_resp = (block_df["pressed"] == "").sum()

    # Per-cue mean RT (correct trials only, in seconds).
    rt_by_cue = {c: _safe_mean(correct.loc[correct["cue"] == c, "rt"])
                 for c in CUE_TYPES}
    rt_by_cong = {c: _safe_mean(correct.loc[correct["congruency"] == c, "rt"])
                  for c in CONGRUENCY}

    # Score formulas (in ms; RT is in seconds).
    def diff_ms(a, b):
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        return (a - b) * 1000.0

    scores = {
        "alerting":  diff_ms(rt_by_cue["NC"], rt_by_cue["DC"]),
        "orienting": diff_ms(rt_by_cue["CC"], rt_by_cue["SC"]),
        "conflict":  diff_ms(rt_by_cong["incongruent"], rt_by_cong["congruent"]),
    }

    return {
        "block": int(block_num),
        "list_letter": list_letter,
        "list_name": f"block_{list_letter}" if list_letter else f"block{block_num}",
        "n_trials": n_trials,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "n_no_resp": int(n_no_resp),
        "accuracy": n_correct / n_trials if n_trials else float("nan"),
        "mean_rt": _safe_mean(correct["rt"]),
        "rt_by_cue": rt_by_cue,
        "rt_by_cong": rt_by_cong,
        "scores": scores,
        "trials": block_df.to_dict("records"),
    }


# --- Aggregation -----------------------------------------------------------


def _diff_of_means_stats(rts_a, rts_b):
    """Return (mean_diff_ms, se_ms, ci_lo_ms, ci_hi_ms) for a − b.

    SE follows the standard error of the difference of two independent
    means: sqrt(var_A/n_A + var_B/n_B). 95% CI uses the normal
    approximation (1.96 · SE) — adequate at typical ANT n per cell
    (~20–60 correct trials) and consistent with how the Bayesian tab
    reports its credible interval.
    """
    a = pd.to_numeric(pd.Series(rts_a), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(rts_b), errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = (a.mean() - b.mean()) * 1000.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) * 1000.0
    return diff, se, diff - 1.96 * se, diff + 1.96 * se


def aggregate_summary(blocks: list[dict], all_trials: pd.DataFrame) -> dict:
    correct = all_trials[all_trials["correct_resp"]]
    rt_by_cue = {c: _safe_mean(correct.loc[correct["cue"] == c, "rt"])
                 for c in CUE_TYPES}
    rt_by_cong = {c: _safe_mean(correct.loc[correct["congruency"] == c, "rt"])
                  for c in CONGRUENCY}

    # Per-score: point estimate, SE, 95% CI from the pooled per-trial
    # variances of the two groups that define the score.
    score_stats = {}
    pairs = {
        "alerting":  (correct.loc[correct["cue"] == "NC", "rt"],
                      correct.loc[correct["cue"] == "DC", "rt"]),
        "orienting": (correct.loc[correct["cue"] == "CC", "rt"],
                      correct.loc[correct["cue"] == "SC", "rt"]),
        "conflict":  (correct.loc[correct["congruency"] == "incongruent", "rt"],
                      correct.loc[correct["congruency"] == "congruent", "rt"]),
    }
    for key, (rts_a, rts_b) in pairs.items():
        diff, se, lo, hi = _diff_of_means_stats(rts_a, rts_b)
        score_stats[key] = {
            "mean": diff, "se": se, "ci_lo": lo, "ci_hi": hi,
            "n_a": int(rts_a.notna().sum()),
            "n_b": int(rts_b.notna().sum()),
        }

    return {
        "n_blocks": len(blocks),
        "n_trials": len(all_trials),
        "n_correct": int(all_trials["correct_resp"].sum()),
        "accuracy": float(all_trials["correct_resp"].mean()),
        "mean_rt": _safe_mean(correct["rt"]),
        "rt_by_cue": rt_by_cue,
        "rt_by_cong": rt_by_cong,
        "scores": {k: v["mean"] for k, v in score_stats.items()},
        "score_stats": score_stats,
    }


# --- Bayesian (analytic Normal-Normal) -------------------------------------


def normal_normal_posterior(prior_mu, prior_sigma, like_sigma, observations):
    """Closed-form posterior for μ in: μ ~ N(μ0, σ0); x_i | μ ~ N(μ, σL).

    Returns (post_mu, post_sigma). Robust to len(observations) == 0
    (returns the prior unchanged) and to NaN observations (filtered out).
    """
    xs = np.asarray([x for x in observations if not math.isnan(x)],
                    dtype=float)
    n = len(xs)
    prec_prior = 1.0 / (prior_sigma ** 2)
    prec_like = n / (like_sigma ** 2) if n else 0.0
    post_var = 1.0 / (prec_prior + prec_like)
    if n:
        post_mu = post_var * (prec_prior * prior_mu
                              + (n / like_sigma ** 2) * xs.mean())
    else:
        post_mu = prior_mu
    return float(post_mu), float(math.sqrt(post_var)), n


def bayesian_summary(blocks: list[dict]) -> dict:
    """Per-score posterior given per-block scores as observations."""
    out = {}
    for key in ("alerting", "orienting", "conflict"):
        obs = [b["scores"][key] for b in blocks]
        prior = PRIOR[key]
        post_mu, post_sigma, n_used = normal_normal_posterior(
            prior["mu"], prior["sigma"], LIKE_SIGMA, obs,
        )
        # 95% credible interval for μ.
        ci_lo = post_mu - 1.96 * post_sigma
        ci_hi = post_mu + 1.96 * post_sigma
        out[key] = {
            "n_obs": n_used,
            "obs": [x for x in obs if not math.isnan(x)],
            "prior": prior,
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    return out


# --- Bokeh helpers ---------------------------------------------------------


def _style_axes(fig):
    fig.toolbar.logo = None
    fig.background_fill_color = "#fafbfc"
    fig.border_fill_color = "white"
    fig.outline_line_color = None
    fig.xgrid.grid_line_color = "#ecf0f1"
    fig.ygrid.grid_line_color = "#ecf0f1"
    fig.axis.axis_line_color = "#bdc3c7"
    fig.axis.major_tick_line_color = "#bdc3c7"
    fig.axis.minor_tick_line_color = None
    fig.axis.axis_label_text_color = "#7f8c8d"
    fig.axis.major_label_text_color = "#7f8c8d"


def block_timeline_figure(block: dict):
    trials = block["trials"]
    if not trials:
        return None

    finite_rts = [t["rt"] for t in trials
                  if isinstance(t["rt"], float) and not math.isnan(t["rt"])]
    y_top = max(finite_rts) * 1.15 if finite_rts else 1.7
    y_floor = -y_top * 0.06

    fig = figure(
        height=240,
        sizing_mode="stretch_width",
        x_axis_label="Trial #",
        y_axis_label="Response time (s)",
        y_range=(y_floor, y_top),
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    _style_axes(fig)

    # One scatter per cue type, hide-toggleable through legend.
    for cue in CUE_TYPES:
        cue_trials = [t for t in trials if t["cue"] == cue]
        if not cue_trials:
            continue
        sub = ColumnDataSource(dict(
            x=[t["trial_idx"] for t in cue_trials],
            y=[t["rt"] if isinstance(t["rt"], float) and not math.isnan(t["rt"])
               else 0 for t in cue_trials],
            cue=[cue] * len(cue_trials),
            target=[t["target"] for t in cue_trials],
            congruency=[t["congruency"] for t in cue_trials],
            correct=[("yes" if t["correct_resp"] else "no") for t in cue_trials],
            rt_str=[(f"{t['rt']:.3f} s"
                     if isinstance(t["rt"], float) and not math.isnan(t["rt"])
                     else "(no press)") for t in cue_trials],
        ))
        # Marker shape encodes correct vs incorrect.
        for outcome, marker in (("yes", "circle"), ("no", "x")):
            mask_idx = [i for i, t in enumerate(cue_trials)
                        if (t["correct_resp"] and outcome == "yes")
                        or (not t["correct_resp"] and outcome == "no")]
            if not mask_idx:
                continue
            mask_source = ColumnDataSource(dict(
                x=[sub.data["x"][i] for i in mask_idx],
                y=[sub.data["y"][i] for i in mask_idx],
                cue=[sub.data["cue"][i] for i in mask_idx],
                target=[sub.data["target"][i] for i in mask_idx],
                congruency=[sub.data["congruency"][i] for i in mask_idx],
                correct=[sub.data["correct"][i] for i in mask_idx],
                rt_str=[sub.data["rt_str"][i] for i in mask_idx],
            ))
            label = f"{cue}" if outcome == "yes" else f"{cue} (error)"
            fig.scatter(
                x="x", y="y", source=mask_source,
                marker=marker, size=10,
                fill_color=CUE_COLOURS[cue],
                line_color=CUE_COLOURS[cue],
                fill_alpha=0.85 if outcome == "yes" else 0.0,
                line_width=2,
                legend_label=label,
            )

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_alpha = 0.85
    fig.legend.label_text_font_size = "10px"
    fig.legend.spacing = 2

    fig.add_tools(HoverTool(tooltips=[
        ("Trial", "@x"),
        ("Cue", "@cue"),
        ("Target", "@target"),
        ("Congruency", "@congruency"),
        ("Correct", "@correct"),
        ("RT", "@rt_str"),
    ]))
    return fig


def rt_distribution_figure(trials: pd.DataFrame, group_col: str,
                           group_order: list, colour_map: dict,
                           title_suffix: str):
    """Histogram + KDE per group, on correct trials only."""
    pressed = trials[(trials["correct_resp"]) & trials["rt"].notna()]
    if pressed.empty:
        return None

    rts = pressed["rt"].to_numpy()
    rt_max = float(np.percentile(rts, 99)) * 1.05
    bins = np.linspace(0, rt_max, 36)
    grid = np.linspace(0, rt_max, 200)

    fig = figure(
        height=320,
        sizing_mode="stretch_width",
        x_axis_label=f"Response time (s) — {title_suffix}",
        y_axis_label="Density",
        x_range=(0, rt_max),
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    _style_axes(fig)

    for group in group_order:
        group_rts = pressed.loc[pressed[group_col] == group, "rt"].to_numpy()
        if len(group_rts) == 0:
            continue
        colour = colour_map.get(group, "#3498db")
        counts, edges = np.histogram(group_rts, bins=bins, density=True)
        fig.quad(
            top=counts, bottom=0,
            left=edges[:-1], right=edges[1:],
            fill_color=colour, fill_alpha=0.18,
            line_color=colour, line_alpha=0.4,
            legend_label=f"{group}  (n={len(group_rts)})",
        )
        if len(group_rts) >= 3 and group_rts.std() > 0:
            kde = stats.gaussian_kde(group_rts)
            density = kde(grid)
            fig.line(grid, density, color=colour, line_width=2.5,
                     legend_label=f"{group}  (n={len(group_rts)})")

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_alpha = 0.85
    return fig


def block_rt_by_cue_figure(block: dict):
    trials = pd.DataFrame(block["trials"])
    return rt_distribution_figure(
        trials, "cue", CUE_TYPES, CUE_COLOURS,
        title_suffix=f"Block {block['block']} — by cue",
    )


def bayesian_density_figure(score_key: str, info: dict):
    """Prior + reference + posterior densities for one score, with the
    individual per-block observations marked as rug ticks at y=0."""
    prior = info["prior"]
    ref = REF_DIST[score_key]
    post_mu = info["post_mu"]
    post_sigma = info["post_sigma"]

    # Build a sensible x-range that covers prior, posterior, reference,
    # and any extreme observations.
    candidates = []
    candidates += [prior["mu"] - 3 * prior["sigma"],
                   prior["mu"] + 3 * prior["sigma"]]
    candidates += [ref["mu"] - 3 * ref["sigma"],
                   ref["mu"] + 3 * ref["sigma"]]
    candidates += [post_mu - 4 * post_sigma, post_mu + 4 * post_sigma]
    candidates += info["obs"]
    x_lo, x_hi = min(candidates) - 10, max(candidates) + 10
    x = np.linspace(x_lo, x_hi, 400)

    def pdf(mu, sigma):
        return stats.norm.pdf(x, mu, sigma)

    y_top = max(pdf(prior["mu"], prior["sigma"]).max(),
                pdf(ref["mu"], ref["sigma"]).max(),
                pdf(post_mu, post_sigma).max()) * 1.1

    fig = figure(
        height=260,
        sizing_mode="stretch_width",
        x_axis_label=f"{SCORE_LABELS[score_key]} score (ms)  =  {SCORE_FORMULA[score_key]}",
        y_axis_label="Density",
        x_range=(x_lo, x_hi),
        y_range=(-y_top * 0.05, y_top),
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    _style_axes(fig)

    colour = SCORE_COLOURS[score_key]

    # Reference (Fan & Posner) — neutral grey, ghosted in back.
    fig.varea(x=x, y1=0, y2=pdf(ref["mu"], ref["sigma"]),
              fill_color="#bdc3c7", fill_alpha=0.18,
              legend_label=f"Fan & Posner ref  μ={ref['mu']}, σ={ref['sigma']}")

    # Prior — score colour at low alpha.
    fig.varea(x=x, y1=0, y2=pdf(prior["mu"], prior["sigma"]),
              fill_color=colour, fill_alpha=0.10,
              legend_label=f"Prior  μ={prior['mu']}, σ={prior['sigma']}")
    fig.line(x, pdf(prior["mu"], prior["sigma"]),
             line_color=colour, line_dash="dashed", line_width=2,
             legend_label=f"Prior  μ={prior['mu']}, σ={prior['sigma']}")

    # Posterior — solid fill + line.
    fig.varea(x=x, y1=0, y2=pdf(post_mu, post_sigma),
              fill_color=colour, fill_alpha=0.30,
              legend_label=f"Posterior  μ={post_mu:.1f}, σ={post_sigma:.1f}")
    fig.line(x, pdf(post_mu, post_sigma),
             line_color=colour, line_width=3,
             legend_label=f"Posterior  μ={post_mu:.1f}, σ={post_sigma:.1f}")

    # Per-block observations as rug ticks at y=0.
    if info["obs"]:
        fig.scatter(
            x=info["obs"], y=[0] * len(info["obs"]),
            marker="dash", size=18,
            line_color=colour, line_width=3,
            legend_label=f"Per-block scores (n={len(info['obs'])})",
        )

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_alpha = 0.85
    fig.legend.label_text_font_size = "10px"
    return fig


# --- HTML rendering --------------------------------------------------------


CSS = """
:root {
  --ink: #2c3e50;
  --muted: #7f8c8d;
  --line: #ecf0f1;
  --card: #ffffff;
  --bg: #fafbfc;
  --accent: #3498db;
  --warn: #e74c3c;
  --pass: #27ae60;
  --fail: #c0392b;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.5;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 1.5em 1em 4em; }
h1 { margin: 0 0 0.2em; font-weight: 600; letter-spacing: -0.01em; }
h2 { margin: 1.6em 0 0.6em; font-weight: 600; letter-spacing: -0.005em; }
h3 { margin: 0 0 0.5em; font-weight: 600; }
.meta { color: var(--muted); font-size: 0.9em; margin: 0 0 1.5em; }
.kv {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.6em 1.2em;
  margin: 0.8em 0 1.2em;
}
.kv > div { font-size: 0.9em; }
.kv .label { display: block; color: var(--muted); font-size: 0.78em;
             text-transform: uppercase; letter-spacing: 0.06em; }
.kv .value { font-weight: 600; font-size: 1.05em; color: var(--ink); }
.tab-bar {
  display: flex; gap: 4px; border-bottom: 2px solid var(--line);
  margin: 1em 0 1.4em; flex-wrap: wrap;
}
.tab-button {
  padding: 10px 20px; border: none; background: none;
  cursor: pointer; font-size: 14px; color: var(--muted);
  border-radius: 4px 4px 0 0; font-family: inherit;
  border-bottom: 3px solid transparent; margin-bottom: -2px;
}
.tab-button:hover { background: var(--line); color: var(--ink); }
.tab-button.active {
  background: var(--card); color: var(--ink);
  border-bottom-color: var(--accent);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 1.1em 1.2em; margin-bottom: 1em;
}
.summary {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1em; align-items: start;
}
@media (max-width: 800px) { .summary { grid-template-columns: 1fr; } }
.score-cards {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8em;
  margin: 0.5em 0 1em;
}
@media (max-width: 700px) { .score-cards { grid-template-columns: 1fr; } }
.score-card {
  padding: 0.9em 1em; border-radius: 10px;
  border: 1px solid var(--line); background: var(--card);
}
.score-card .name {
  font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 0.3em;
}
.score-card .value {
  font-size: 1.7em; font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.score-card .value-sd {
  font-size: 0.5em; font-weight: 500; color: var(--muted);
  margin-left: 0.3em; letter-spacing: 0.01em;
}
.score-card .ci {
  font-size: 0.78em; color: var(--muted); margin-top: 0.15em;
  font-variant-numeric: tabular-nums;
}
.score-card .formula { font-size: 0.75em; color: var(--muted); margin-top: 0.3em; }
.score-card .ref { font-size: 0.75em; color: var(--muted); margin-top: 0.3em; }
table.report {
  border-collapse: collapse; width: 100%; font-size: 0.92em;
}
table.report th, table.report td {
  padding: 0.45em 0.7em; border-bottom: 1px solid var(--line); text-align: left;
}
table.report th {
  background: var(--bg); color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.75em;
}
table.report tbody tr:hover { background: var(--bg); }
table.report .num { text-align: right; font-variant-numeric: tabular-nums; }
.block-tag {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 600; font-size: 0.95em;
  padding: 0.15em 0.55em; border-radius: 6px; background: var(--bg);
}
.legend { font-size: 0.85em; color: var(--muted); margin: 0.4em 0 0.8em;
          display: flex; flex-wrap: wrap; gap: 0.8em; }
.legend .swatch { display: inline-block; width: 12px; height: 12px;
                  border-radius: 3px; vertical-align: middle; margin-right: 4px; }
.score-pill {
  display: inline-block; padding: 0.05em 0.5em; border-radius: 999px;
  font-size: 0.78em; font-weight: 600; font-variant-numeric: tabular-nums;
}
.back-link { font-size: 0.9em; }
.back-link a { color: var(--accent); text-decoration: none; }
.back-link a:hover { text-decoration: underline; }
.cue-chip {
  display: inline-block; padding: 0.05em 0.45em; border-radius: 5px;
  font-size: 0.8em; color: white; font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.note { color: var(--muted); font-size: 0.85em; }
hr { border: none; border-top: 1px solid var(--line); margin: 1.5em 0; }
"""

JS = """
document.addEventListener('click', function (e) {
  if (!e.target.matches('.tab-button')) return;
  var tab = e.target.dataset.tab;
  document.querySelectorAll('.tab-button').forEach(function (b) {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  document.querySelectorAll('.tab-panel').forEach(function (p) {
    p.classList.toggle('active', p.id === 'tab-' + tab);
  });
  if (history.replaceState) {
    history.replaceState(null, null, '#' + tab);
  }
});
window.addEventListener('DOMContentLoaded', function () {
  var hash = (location.hash || '').replace('#', '');
  if (!hash) return;
  var btn = document.querySelector('.tab-button[data-tab="' + hash + '"]');
  if (btn) btn.click();
});
"""


def _fmt_ms(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.0f} ms"


def _fmt_rt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f} s"


def _fmt_pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.1f}%"


def _score_pill(value, key):
    colour = SCORE_COLOURS[key]
    return (f'<span class="score-pill" '
            f'style="background:{colour}22;color:{colour};">'
            f'{html.escape(_fmt_ms(value))}</span>')


def _kv_html(items):
    return "".join(
        f'<div><span class="label">{html.escape(k)}</span>'
        f'<span class="value">{html.escape(str(v))}</span></div>'
        for k, v in items
    )


def _legend_cues_html():
    swatches = "".join(
        f'<span><span class="swatch" style="background:{CUE_COLOURS[c]}"></span>'
        f'{c} <span class="note">{CUE_LABEL[c]}</span></span>'
        for c in CUE_TYPES
    )
    return f'<div class="legend">{swatches}</div>'


def _legend_cong_html():
    swatches = "".join(
        f'<span><span class="swatch" style="background:{CONGRUENCY_COLOURS[c]}"></span>{c}</span>'
        for c in CONGRUENCY
    )
    return f'<div class="legend">{swatches}</div>'


def render_overview_tab(run, agg, blocks_table_html, by_cue_table_html,
                        by_cong_table_html):
    def _card(k):
        st = agg["score_stats"][k]
        sd_str = (f"&plusmn; {st['se']:.0f} ms"
                  if not math.isnan(st["se"]) else "")
        ci_str = (f"95% CI [{st['ci_lo']:+.0f}, {st['ci_hi']:+.0f}] ms"
                  if not math.isnan(st["ci_lo"]) else "—")
        return (
            f'<div class="score-card" style="border-left:4px solid {SCORE_COLOURS[k]}">'
            f'<div class="name">{SCORE_LABELS[k]}</div>'
            f'<div class="value" style="color:{SCORE_COLOURS[k]}">'
            f'{html.escape(_fmt_ms(agg["scores"][k]))}'
            f'<span class="value-sd"> {sd_str}</span></div>'
            f'<div class="ci">{ci_str}</div>'
            f'<div class="formula">{html.escape(SCORE_FORMULA[k])}</div>'
            f'<div class="ref">Fan&amp;Posner ref: μ={REF_DIST[k]["mu"]:.0f}, '
            f'σ={REF_DIST[k]["sigma"]:.0f} ms</div>'
            f"</div>"
        )
    score_cards = "".join(_card(k) for k in ("alerting", "orienting", "conflict"))

    head_kv = [
        ("Participant", run["participant"]),
        ("Session", run["session"] or "—"),
        ("Date", run["date_str"] or "—"),
        ("Duration", _format_duration(run["duration_s"])),
        ("Blocks ran", str(agg["n_blocks"])),
        ("nBlocks setting", run["n_blocks_setting"] or "—"),
        ("Trials", str(agg["n_trials"])),
        ("Accuracy", _fmt_pct(agg["accuracy"])),
        ("Mean RT", _fmt_rt(agg["mean_rt"])),
        ("PsychoPy", run["psychopy_version"] or "—"),
    ]

    return f"""
    <div class="tab-panel active" id="tab-overview">
      <div class="kv">{_kv_html(head_kv)}</div>

      <h2>Aggregate scores</h2>
      <p class="note">Computed on correct trials only across {agg['n_blocks']}
      block(s); positive scores are interpreted as the network being engaged
      (slower RT in the harder condition). The &plusmn; figure is the
      standard error of the mean difference; the 95% CI uses the normal
      approximation. For Bayesian credible intervals (CrI) updated against
      literature priors, see the <a href="#bayesian">Bayesian</a> tab.</p>
      <div class="score-cards">{score_cards}</div>

      <h2>Per-block scores</h2>
      {_legend_cues_html()}
      {blocks_table_html}

      <div class="summary" style="margin-top:1.5em">
        <div class="card">
          <h3>Mean RT by cue (correct trials)</h3>
          {by_cue_table_html}
        </div>
        <div class="card">
          <h3>Mean RT by congruency (correct trials)</h3>
          {_legend_cong_html()}
          {by_cong_table_html}
        </div>
      </div>
    </div>
    """


def render_perblock_tab(blocks, block_timeline_divs, block_hist_divs):
    cards = []
    for b in blocks:
        scores_html = " &middot; ".join(
            f'{SCORE_LABELS[k]} {_score_pill(b["scores"][k], k)}'
            for k in ("alerting", "orienting", "conflict")
        )
        per_cue_rt = " &middot; ".join(
            f'<span class="cue-chip" style="background:{CUE_COLOURS[c]}">{c}</span> '
            f'{_fmt_rt(b["rt_by_cue"][c])}'
            for c in CUE_TYPES
        )
        timeline = block_timeline_divs.get(b["block"], "")
        hist = block_hist_divs.get(b["block"], "")
        cards.append(f"""
        <article class="card">
          <div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:0.8em;margin-bottom:0.4em">
            <span class="block-tag">Block {b['block']} &middot; {html.escape(b['list_name'])}.csv</span>
            <span class="note">{b['n_trials']} trials &middot; accuracy
              <b>{_fmt_pct(b['accuracy'])}</b> &middot; mean RT
              <b>{_fmt_rt(b['mean_rt'])}</b></span>
          </div>
          <div style="margin:0.4em 0 0.6em">{scores_html}</div>
          <div class="note" style="margin-bottom:0.5em">{per_cue_rt}</div>
          <h3 style="margin-top:0.8em">Timeline</h3>
          {timeline}
          <h3 style="margin-top:0.8em">RT distribution by cue</h3>
          {hist}
        </article>
        """)
    return f"""
    <div class="tab-panel" id="tab-perblock">
      {_legend_cues_html()}
      {''.join(cards) if cards else '<p class="note">No blocks.</p>'}
    </div>
    """


def render_distributions_tab(rt_by_cue_div, rt_by_cong_div):
    return f"""
    <div class="tab-panel" id="tab-distributions">
      <div class="card">
        <h3>RT distribution by cue type</h3>
        {_legend_cues_html()}
        <p class="note">Histogram + Gaussian KDE per cue type, correct trials
        only. Click a legend entry to hide that series.</p>
        {rt_by_cue_div or '<p class="note">No data.</p>'}
      </div>
      <div class="card">
        <h3>RT distribution by flanker congruency</h3>
        {_legend_cong_html()}
        <p class="note">Same data grouped by the central-arrow's flanker
        congruency. Neutral trials use the dashed neutral target.</p>
        {rt_by_cong_div or '<p class="note">No data.</p>'}
      </div>
    </div>
    """


def render_bayesian_tab(bayes, bayes_divs, n_blocks):
    rows = []
    for k in ("alerting", "orienting", "conflict"):
        info = bayes[k]
        prior = info["prior"]
        rows.append(
            f'<tr>'
            f'<td><span style="color:{SCORE_COLOURS[k]};font-weight:600">'
            f'{SCORE_LABELS[k]}</span></td>'
            f'<td class="num">{info["n_obs"]}</td>'
            f'<td class="num">{prior["mu"]:.0f} ± {prior["sigma"]:.0f}</td>'
            f'<td class="num">{REF_DIST[k]["mu"]:.0f} ± {REF_DIST[k]["sigma"]:.0f}</td>'
            f'<td class="num">{info["post_mu"]:.1f} ± {info["post_sigma"]:.1f}</td>'
            f'<td class="num">[{info["ci_lo"]:.1f}, {info["ci_hi"]:.1f}]</td>'
            f'</tr>'
        )

    n_blocks_caveat = ""
    if n_blocks <= 1:
        n_blocks_caveat = (
            '<p class="note"><strong>Caveat:</strong> with only one block of '
            'data, the per-score posterior is dominated by the prior. Run '
            'more blocks to let the data update beliefs more strongly.</p>'
        )

    figs_html = "".join(
        f'<div class="card"><h3 style="color:{SCORE_COLOURS[k]}">'
        f'{SCORE_LABELS[k]}: {SCORE_FORMULA[k]}</h3>{bayes_divs[k]}</div>'
        for k in ("alerting", "orienting", "conflict")
    )

    return f"""
    <div class="tab-panel" id="tab-bayesian">
      <div class="card">
        <h3>Posterior summary</h3>
        <p class="note">Closed-form Normal–Normal conjugate update:
        prior on the participant's mean μ for each ANT score, then per-block
        scores treated as observations <code>x_i ~ Normal(μ, σ_L)</code> with
        σ_L = {LIKE_SIGMA:.0f} ms. The posterior is analytic, no MCMC needed.
        Priors come from the per-block Turing model in the previous Julia
        analysis (centred on Fan&amp;Posner-style population means with σ=100 ms,
        deliberately broad). Reference column is the population distribution
        from Fan &amp; Posner.</p>
        <table class="report">
          <thead><tr>
            <th>Score</th><th class="num">Blocks (n)</th>
            <th class="num">Prior μ ± σ (ms)</th>
            <th class="num">Reference μ ± σ (ms)</th>
            <th class="num">Posterior μ ± σ (ms)</th>
            <th class="num">95% CrI for μ</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        {n_blocks_caveat}
      </div>
      {figs_html}
    </div>
    """


def render_report(run: dict, out_path: Path) -> None:
    blocks = run["blocks"]
    agg = aggregate_summary(blocks, run["trials"])
    bayes = bayesian_summary(blocks)

    # Prepare all Bokeh figures up-front so we get one components() pass
    # (one <script> block, divs in stable order).
    fig_objects = []
    fig_keys = []  # parallel list of identifiers to map back to divs

    for b in blocks:
        f = block_timeline_figure(b)
        if f is not None:
            fig_objects.append(f)
            fig_keys.append(("timeline", b["block"]))
        f = block_rt_by_cue_figure(b)
        if f is not None:
            fig_objects.append(f)
            fig_keys.append(("hist", b["block"]))

    rt_cue_fig = rt_distribution_figure(
        run["trials"], "cue", CUE_TYPES, CUE_COLOURS, "by cue type",
    )
    if rt_cue_fig is not None:
        fig_objects.append(rt_cue_fig)
        fig_keys.append(("rt_cue", None))

    rt_cong_fig = rt_distribution_figure(
        run["trials"], "congruency", CONGRUENCY, CONGRUENCY_COLOURS,
        "by flanker congruency",
    )
    if rt_cong_fig is not None:
        fig_objects.append(rt_cong_fig)
        fig_keys.append(("rt_cong", None))

    for k in ("alerting", "orienting", "conflict"):
        f = bayesian_density_figure(k, bayes[k])
        fig_objects.append(f)
        fig_keys.append(("bayes", k))

    if fig_objects:
        script, divs = components(fig_objects)
    else:
        script, divs = "", []

    div_by_key = dict(zip(fig_keys, divs))
    block_timeline_divs = {key[1]: d for key, d in div_by_key.items()
                           if key[0] == "timeline"}
    block_hist_divs = {key[1]: d for key, d in div_by_key.items()
                       if key[0] == "hist"}
    rt_cue_div = div_by_key.get(("rt_cue", None), "")
    rt_cong_div = div_by_key.get(("rt_cong", None), "")
    bayes_divs = {k: div_by_key[("bayes", k)]
                  for k in ("alerting", "orienting", "conflict")}

    # Per-block table on Overview tab.
    block_rows = []
    for b in blocks:
        scores_cells = "".join(
            f'<td class="num">{_score_pill(b["scores"][k], k)}</td>'
            for k in ("alerting", "orienting", "conflict")
        )
        block_rows.append(
            f"<tr>"
            f"<td><b>Block {b['block']}</b> "
            f'<span class="note">{html.escape(b["list_name"])}.csv</span></td>'
            f'<td class="num">{b["n_trials"]}</td>'
            f'<td class="num">{_fmt_pct(b["accuracy"])}</td>'
            f'<td class="num">{_fmt_rt(b["mean_rt"])}</td>'
            f"{scores_cells}"
            f"</tr>"
        )
    aggregate_row = (
        f'<tr style="font-weight:600;background:var(--bg)">'
        f'<td>Aggregate</td>'
        f'<td class="num">{agg["n_trials"]}</td>'
        f'<td class="num">{_fmt_pct(agg["accuracy"])}</td>'
        f'<td class="num">{_fmt_rt(agg["mean_rt"])}</td>'
        f'<td class="num">{_score_pill(agg["scores"]["alerting"], "alerting")}</td>'
        f'<td class="num">{_score_pill(agg["scores"]["orienting"], "orienting")}</td>'
        f'<td class="num">{_score_pill(agg["scores"]["conflict"], "conflict")}</td>'
        f'</tr>'
    )
    blocks_table_html = f"""
    <table class="report">
      <thead><tr>
        <th>Block</th>
        <th class="num">Trials</th>
        <th class="num">Accuracy</th>
        <th class="num">Mean RT</th>
        <th class="num" style="color:{SCORE_COLOURS['alerting']}">Alerting</th>
        <th class="num" style="color:{SCORE_COLOURS['orienting']}">Orienting</th>
        <th class="num" style="color:{SCORE_COLOURS['conflict']}">Conflict</th>
      </tr></thead>
      <tbody>{''.join(block_rows)}{aggregate_row}</tbody>
    </table>
    """

    # By-cue / by-cong tables (aggregate).
    by_cue_table_html = (
        '<table class="report"><thead><tr><th>Cue</th>'
        '<th class="num">Mean RT</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td><span class="cue-chip" '
            f'style="background:{CUE_COLOURS[c]}">{c}</span> '
            f'<span class="note">{CUE_LABEL[c]}</span></td>'
            f'<td class="num">{_fmt_rt(agg["rt_by_cue"][c])}</td></tr>'
            for c in CUE_TYPES
        )
        + "</tbody></table>"
    )
    by_cong_table_html = (
        '<table class="report"><thead><tr><th>Congruency</th>'
        '<th class="num">Mean RT</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td><span class="swatch" '
            f'style="background:{CONGRUENCY_COLOURS[c]};display:inline-block;'
            f'width:10px;height:10px;border-radius:2px;margin-right:6px"></span>'
            f'{c}</td>'
            f'<td class="num">{_fmt_rt(agg["rt_by_cong"][c])}</td></tr>'
            for c in CONGRUENCY
        )
        + "</tbody></table>"
    )

    overview_html = render_overview_tab(
        run, agg, blocks_table_html, by_cue_table_html, by_cong_table_html,
    )
    perblock_html = render_perblock_tab(blocks, block_timeline_divs,
                                        block_hist_divs)
    distributions_html = render_distributions_tab(rt_cue_div, rt_cong_div)
    bayesian_html = render_bayesian_tab(bayes, bayes_divs, len(blocks))

    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(run['participant'])} — ANT report</title>
{INLINE.render()}
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <p class="back-link"><a href="../index.html">&larr; All reports</a></p>
  <h1>{html.escape(run['participant'])}
    <span style="color:var(--muted);font-weight:400">&middot;
    {html.escape(run['date_str'] or '')}</span></h1>
  <p class="meta">{html.escape(run['csv_name'])}</p>

  <div class="tab-bar">
    <button class="tab-button active" data-tab="overview">Overview</button>
    <button class="tab-button" data-tab="perblock">Per-block ({len(blocks)})</button>
    <button class="tab-button" data-tab="distributions">RT distributions</button>
    <button class="tab-button" data-tab="bayesian">Bayesian</button>
  </div>

  {overview_html}
  {perblock_html}
  {distributions_html}
  {bayesian_html}
</div>
{script}
<script>{JS}</script>
<footer style="text-align:center;color:var(--muted);font-size:0.8em;padding:2em 1em">
Generated {html.escape(timestamp)} by report.py
</footer>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)


# --- Index -----------------------------------------------------------------


def render_index(runs: list[dict], index_path: Path) -> None:
    runs_sorted = sorted(
        runs, key=lambda r: r.get("date_str") or "", reverse=True,
    )

    rows = []
    for r in runs_sorted:
        href = f"reports/{r['report_slug']}.html"
        agg = aggregate_summary(r["blocks"], r["trials"])
        rows.append(
            f'<tr>'
            f'<td><a href="{html.escape(href)}" '
            f'style="color:var(--accent);text-decoration:none">'
            f'{html.escape(r["participant"])}</a></td>'
            f'<td>{html.escape(r["session"] or "—")}</td>'
            f'<td>{html.escape(r["date_str"] or "")}</td>'
            f'<td class="num">{_format_duration(r["duration_s"])}</td>'
            f'<td class="num">{agg["n_blocks"]}</td>'
            f'<td class="num">{_fmt_pct(agg["accuracy"])}</td>'
            f'<td class="num">{_score_pill(agg["scores"]["alerting"], "alerting")}</td>'
            f'<td class="num">{_score_pill(agg["scores"]["orienting"], "orienting")}</td>'
            f'<td class="num">{_score_pill(agg["scores"]["conflict"], "conflict")}</td>'
            f'<td><a href="{html.escape(href)}" '
            f'style="color:var(--accent);text-decoration:none">open &rarr;</a></td>'
            f'</tr>'
        )

    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    body_rows = (
        "".join(rows) if rows
        else '<tr><td colspan="10" style="text-align:center;color:var(--muted);'
             'padding:2em">No reports yet. Run the experiment, then '
             '<code>nix run .#report</code>.</td></tr>'
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANT participant reports</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>ANT participant reports</h1>
  <p class="meta">{len(runs_sorted)} run(s) &middot;
    <a href="report.html" style="color:var(--accent)">prerandomization report &rarr;</a>
  </p>
  <div class="card" style="padding:0;overflow:hidden">
    <table class="report">
      <thead><tr>
        <th>Participant</th><th>Session</th><th>Date</th>
        <th class="num">Duration</th>
        <th class="num">Blocks</th>
        <th class="num">Accuracy</th>
        <th class="num" style="color:{SCORE_COLOURS['alerting']}">Alerting</th>
        <th class="num" style="color:{SCORE_COLOURS['orienting']}">Orienting</th>
        <th class="num" style="color:{SCORE_COLOURS['conflict']}">Conflict</th>
        <th></th>
      </tr></thead>
      <tbody>{body_rows}</tbody>
    </table>
  </div>
</div>
<footer style="text-align:center;color:var(--muted);font-size:0.8em;padding:2em 1em">
Index updated {html.escape(timestamp)}
</footer>
</body>
</html>
"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)


# --- Main ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR,
                        help="data directory containing PsychoPy CSVs")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR,
                        help="output directory for per-participant reports")
    parser.add_argument("--index", type=Path, default=INDEX_PATH,
                        help="path to write the index HTML")
    parser.add_argument("--csv", type=Path, action="append",
                        help="generate a report for this CSV only "
                             "(may be passed multiple times)")
    args = parser.parse_args()

    csv_paths = (args.csv if args.csv else sorted(args.data.glob("*.csv")))
    if not csv_paths:
        print(f"No CSVs in {args.data}")
        return

    runs = []
    skipped = []
    for path in csv_paths:
        try:
            run = load_run(path)
        except Exception as exc:
            print(f"  ! failed to load {path.name}: {exc}")
            continue
        if run is None:
            skipped.append(path.name)
            continue
        out_path = args.reports / f"{run['report_slug']}.html"
        render_report(run, out_path)
        runs.append(run)
        print(f"  + {path.name} -> {out_path.relative_to(out_path.parent.parent)}")

    if skipped:
        print(f"  - skipped {len(skipped)} CSV(s) with no trials")

    if args.csv:
        # Subset run: rebuild full index from every CSV on disk.
        all_runs = []
        for p in sorted(args.data.glob("*.csv")):
            try:
                r = load_run(p)
            except Exception:
                continue
            if r is not None:
                all_runs.append(r)
        render_index(all_runs, args.index)
    else:
        render_index(runs, args.index)
    print(f"  index -> {args.index}")


if __name__ == "__main__":
    main()
