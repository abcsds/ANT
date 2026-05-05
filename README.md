# Attentional Networks Task

ANT for measuring three attentional networks — alerting, orienting, and
executive control — implemented in PsychoPy. Spanish-language instructions.
Originally described in Fan et al. (2002) [^1], adapted following van Vugt &
van den Hurk (2017) [^2].

[^1]: Fan, J., McCandliss, B. D., Sommer, T., Raz, A., & Posner, M. I. (2002).
    Testing the efficiency and independence of attentional networks. *Journal
    of Cognitive Neuroscience*, 14(3), 340–347.
[^2]: van Vugt, M. K., & van den Hurk, P. M. (2017). Modeling the effects of
    attentional cueing on meditators. *Mindfulness*, 8, 38–45.

## Parameters

The numbers below are defined in `prerandomize.py` and mirrored in
`ANT.psyexp`. Change them in one place and the other.

| Parameter | Value |
| --- | --- |
| Fixation duration (D1) | **0.4 + uniform(0, 1.2) s** &rarr; mean **1.0 s** |
| Cue routine duration | **0.4 s** total, cue stim on-screen for the first **0.1 s** |
| Cue stim size | letterHeight **0.15** (asterisk, white) — clearly larger than the 0.05 fixation cross |
| Cue draw order | the asterisk is drawn **on top of** the cross (`fixation_cross` enters the window's autoDraw list at block start; `text_1_cue` / `text_2_cue` are appended later when the Cue routine begins, so they render after the cross) |
| Fixation cross | persistent for the entire block (single `setAutoDraw` stim, **not** redrawn per trial) |
| Flanker font | **DejaVu Sans Mono** so `<`, `>`, and `-` have identical advance widths (essential for the neutral `--<--` / `-->--` flankers to align with `<<<<<` / `>>>>>`) |
| Target max duration | **1.7 s** (or earlier on `f` / `j` keypress) |
| Assumed mean RT | **0.6 s** (used only to estimate task duration) |
| Mean trial duration | **2.0 s** |
| Trials per block | **120** (4 cues &times; 6 targets &times; 5 reps) |
| Block lists | **10** (`block_a.csv` &hellip; `block_j.csv`) |
| Max blocks per run | **10** (each list used at most once per participant) |
| Inter-block break | **60 s**, self-paced |
| Response keys | `f` (left), `j` (right) |
| Target offset from fixation | ±0.25 height units (above / below) |
| Window | 1920 × 1080, fullscreen (windowed in pilot mode) |

A *trial* is `Fix → Cue → Target`; a *block* is one CSV from `lists/`.

## Cue conditions (`cue` column)

Each trial uses one of four cues, flashed for 100 ms inside the 400 ms cue
routine. The fixation cross stays on screen for the full 400 ms.

| Code | Cue type | Behaviour |
| --- | --- | --- |
| `NC` | None     | No cue text shown (cue text positioned far off-screen) |
| `CC` | Central  | Single `*` at fixation |
| `DC` | Double   | Two `*` simultaneously above and below fixation |
| `SC` | Spatial  | Single `*` above OR below fixation (chosen at random), predicting where the target will appear |

## Target conditions (`target` column)

Five-arrow flanker; the participant responds to the **central** arrow only.

| Stimulus  | Direction | Flanker     | Correct key |
| ---       | ---       | ---         | ---         |
| `<<<<<`   | Left      | Congruent   | `f` |
| `>>>>>`   | Right     | Congruent   | `j` |
| `>><>>`   | Left      | Incongruent | `f` |
| `<<><<`   | Right     | Incongruent | `j` |
| `--<--`   | Left      | Neutral     | `f` |
| `-->--`   | Right     | Neutral     | `j` |

On `SC` trials the target appears at the cued position; on `NC` / `CC` / `DC`
trials it appears above or below fixation with equal probability.

## Pre-randomization

Trial sequences are generated ahead of time and stored as CSVs in `lists/`,
one row per trial:

```csv
cue,target,correct
DC,<<><<,j
NC,--<--,f
...
```

Ten such block lists are generated (`block_a.csv` &hellip; `block_j.csv`),
each containing **120 trials** = 5 copies of every (cue, target)
condition in a random order. Within each block every cue appears 30 times
and `f`/`j` each appear 60 times by construction.

### Participant schedules

`prerandomize.py` also emits 1000 per-participant schedule files in
`schedules/`, named `000.csv` … `999.csv` — one for every 3-digit
participant code. Each schedule has 10 rows (the 10 block-list letters
permuted once) and looks like:

```csv
block,list_letter,condsFile
1,a,lists/block_a.csv
2,f,lists/block_f.csv
...
10,h,lists/block_h.csv
```

At runtime the experiment loads `schedules/{participant}.csv`, walks it
sequentially, and stops as soon as `block > nBlocks`. With the same
`DEFAULT_SEED`, all 1000 schedule files are byte-identical.

### Generating the lists

With Nix:

```sh
nix run .#prerandomize
```

Or directly (no third-party dependencies — stdlib only):

```sh
python3 prerandomize.py [--seed N] [--output lists] [--schedules schedules] [--log LOG.md] [--report docs/report.html]
```

The script:

1. Builds 10 block lists of 120 trials each, asserting that every (cue,
   target) condition appears exactly 5 times and that `f`/`j` correct
   counts are balanced.
2. Builds 1000 participant schedules, asserting that every list-letter
   appears exactly once per schedule and rows are sorted by block ascending.
3. Renders `docs/report.html` (see below).
4. Compares the seed used against the most recent entry in `LOG.md` and
   appends a new line (timestamp + seed) only if the seed has changed.

### HTML report

Every run also writes `docs/report.html` (single page, no external
dependencies). It shows:

- **Timing parameters** — every constant from `prerandomize.py` so you
  can confirm what the experiment will actually do.
- **Experiment schedule** — wall-clock estimate as a function of
  `nBlocks` (1…10): instructions duration + task time (blocks + breaks)
  + total. Useful for planning sessions.
- **Sample participant schedule** — the contents of `schedules/000.csv`,
  rendered as a table, so you can sanity-check the format.
- **Aggregate counts across all 10 lists** — bar plots of cue type,
  flanker congruency, and target arrow. All should be flat by
  construction.
- **Per-list timelines** — one row per block list, one cell per trial.
  Cell fill encodes cue type (NC / CC / DC / SC), the top stripe encodes
  flanker congruency, and the arrow inside encodes target direction.

Open `docs/report.html` in any browser to review the lists before running
the experiment.

### Reproducibility & seed

The default seed is set at the top of `prerandomize.py` (constant
`DEFAULT_SEED`). Running the script with the same seed always produces
byte-identical CSV files. To regenerate with a new randomization, change
`DEFAULT_SEED` (or pass `--seed`) and re-run; `LOG.md` will be appended
with the new seed and the run timestamp. `LOG.md` is the audit trail:
the most recent entry identifies which seed produced the current contents
of `lists/` and `schedules/`.

## Implementation notes

The PsychoPy experiment (`ANT.psyexp`) consumes the pre-randomized
files. The participant-info dialog has two relevant fields:

| Field | Range | Purpose |
| --- | --- | --- |
| `participant` | `000`…`999` (or any string) | Numeric IDs map directly to `schedules/{ID}.csv`. Non-numeric IDs (e.g. `pilot`) are MD5-hashed deterministically to a 3-digit code |
| `nBlocks` | 1…10 | Number of blocks the participant will run |

At experiment start, `code_welcome` builds
`schedules/{participant_id:03d}.csv`, asserts the file exists, and
initialises `block = 0`. The outer `scheduleLoop` then reads the 10 rows
sequentially. The inner `trials` loop's `conditionsFile` is set per-row
via `$condsFile`, and its `nReps` is `1 if block <= n_blocks else 0`, so
schedule rows past `nBlocks` skip the trial loop entirely. The
`BlockEnd` break routine has the matching `skipIf = block > n_blocks`,
so the participant only sees a break message after blocks they actually
ran.

### Trial structure

```
Fix  (0.4 – 1.6 s)
 → Cue (0.4 s, cue stim during first 0.1 s)
 → Target (≤ 1.7 s; ends on keypress)
```

The fixation cross is drawn **once per block** (turned on at the first
trial of each block via `setAutoDraw=True` in `code_fix`, turned off in
`code_blockend`) and stays visible continuously throughout the block.
This avoids the 1–2-frame gaps that would appear if the cross were
re-drawn in every routine, so the cross's appearance is not itself a
per-trial temporal cue — only the jittered fixation duration is.

The flow is `Welcome → Intro → 10× scheduleLoop[trials → BlockEnd] →
End`, with the schedule loop's row count effectively capped at `nBlocks`
(rows beyond that run the empty inner loop and skip BlockEnd).
The break message after each block reads **"Este es el final del bloque
{block} de {n_blocks}."** so the participant always sees their progress.

## Per-participant reports

After every experiment run, build per-participant HTML reports from each
PsychoPy CSV in `data/`:

```sh
nix run .#report
```

This scans every `data/*.csv`, writes `docs/reports/<csv-stem>.html` for
each finished run, and rebuilds `docs/index.html` so it lists every
available report newest-first. Old-schema CSVs (without the new
`block` / `list_letter` / `condsFile` columns) are skipped silently.

Each participant report has four tabs:

- **Overview** — run metadata, aggregate Alerting / Orienting / Conflict
  cards (with reference distributions from Fan & Posner), per-block
  scores table, and aggregate mean RT broken down by cue type and by
  flanker congruency. Scores are computed on **correct trials only**:

  - **Alerting** = `RT(NC) − RT(DC)` (no cue minus double cue)
  - **Orienting** = `RT(CC) − RT(SC)` (centre cue minus spatial cue)
  - **Conflict** = `RT(IC) − RT(CG)` (incongruent minus congruent flankers)

- **Per-block** — one card per block: score row, interactive Bokeh
  timeline (RT per trial coloured by cue, hover gives target + congruency
  + correctness, errors marked with an X), and a per-cue RT histogram.
- **RT distributions** — Bokeh histogram + KDE overlays of every correct
  RT, one figure coloured by cue type and one by flanker congruency.
  Click a legend entry to hide a series.
- **Bayesian** — closed-form Normal–Normal conjugate posterior for each
  score given Fan & Posner literature priors (μ_A=40, μ_O=50, μ_C=98 ms;
  σ=100 ms throughout). Per-block scores are treated as observations.
  The page shows the prior–reference–posterior density overlay per score,
  posterior summary table (μ, σ, 95% CrI), and individual block scores
  rugged at y=0.

Reports are self-contained (Bokeh JS/CSS inlined) so they open offline,
and the visual style matches `docs/report.html` (the prerandomization
report) and the N-back report site.

