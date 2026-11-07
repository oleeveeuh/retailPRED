# Portfolio entry generator — paste into any project's session

You are working in a project repository. The goal: produce **one complete
portfolio entry** on [oleeveeuh.github.io](https://oleeveeuh.github.io) —
the `projects.js` block, the card photo (a method schematic), the story
figures, and the dialog story — matching the `retailpred` reference entry.
Scope ends there: **no showcase website, no new pages, no site restyling.**
The entry, its figures, and its story are the whole deliverable. There is
nothing to deploy per project; the entry lives on the existing site.

## Read first

- **This project's frozen artifacts** — committed results JSONs, frozen
  docs, the README's own measured claims. Every number in the entry must be
  copied from one of these, and you must be able to name the file it came
  from. A project with no frozen artifacts gets a placeholder entry: no
  `previewImage`, no figures, no numbers (the site renders placeholder SVGs
  for entries without figures — that is the honest state, not a failure).
- **`~/portfolio-site/PROMPT_PROJECT_ENTRIES.md`** — authoritative for site
  mechanics: entry schema, story block grammar, renderer behavior, skill
  groups, categorization, pitfalls. Follow it; this prompt adds the settled
  styling and the workflow around it.
- **`~/portfolio-site/projects.js`** — find `id: "retailpred"`. That entry
  is the reference shape: copy its structure exactly, do not invent.
- **`~/portfolio-site/app.js`** — renderers, `techMark()`, `artSVG()`.

## The styling is settled — copy it, never re-derive it

Fonts are installed locally as static TTFs in `~/Library/Fonts` (Source
Sans 3 400/600/700; IBM Plex Mono 400/500). Verify before rendering:

```python
from matplotlib import font_manager as fm
fm.findfont(fm.FontProperties(family="Source Sans 3"), fallback_to_default=False)
```

```python
# figure rcParams — every figure script in the house style starts here
plt.rcParams["font.sans-serif"] = ["Source Sans 3", "Helvetica Neue", "Arial",
                                   "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["IBM Plex Mono", "Courier New",
                                  "DejaVu Sans Mono"]
plt.rcParams["font.family"] = "sans-serif"
```

- **Source Sans 3** is the body voice: all primary figure text.
- **IBM Plex Mono** is the caption voice: footnotes, cron/cadence lines,
  annotation eyebrows, artifact labels under nodes.
- matplotlib has no letter-spacing. Headings mimic the site's
  letter-spaced mono eyebrows with space-joined caps:
  `" ".join("THE TITLE")` at 14pt.
- Traps: matplotlib **silently falls back to DejaVu Sans** when a family is
  missing — check `findfont`, don't trust the render. macOS Helvetica Neue
  is a `.ttc` matplotlib cannot load at all; never first in a stack.

Palette (the fieldnotes theme — the whole site reads as one system):

```python
SURFACE = "#e4e4d7"   # card-art light green — the preview's background
PAPER   = "#fbf9f4"   # cream card surface
INK     = "#25251f"
INK2    = "#52514e"
MUTED   = "#77776c"
GRID    = "#d8d1c4"
SAGE    = "#89927a"   # positive / site accent
SAGE_LIGHT = "#dfe1d5"
WINE    = "#482728"   # failure / refusal voice
CRIMSON = "#a43b36"   # site red accent — negatives, links
SUMI    = "#1E342E"   # neutral baseline (deepest tone)
ROSE    = "#d4708c"   # fourth identity hue
```

Voice mapping: **sage = positive**, **wine/crimson = failure, refusal,
negative results** — negative results are headline material, never buried.
Never place rose beside wine or crimson; never introduce a fifth identity
hue; fold extras into labels.

**Card canvas (the trap that cost sessions):** the gallery art box is
320.7×160 (aspect ≈ 2.004) and shows previews with `object-fit: cover`.
Render with `bbox_inches="tight"`, then **pad the canvas to the exact
aspect with the surface color** — the wings are invisible because the
padding color IS the card background. Copy the save block verbatim from
`~/retailPRED-satellite/scripts/method_schematic.py` (the `io.BytesIO` →
`plt.imread` → wing-padding → `plt.imsave` block). `figsize ≈ (6.42, 3.3)`;
type 7–16pt; `pad_inches=0.2`.

## Deliverable 1 — the card photo: a method schematic

The card is the project's hook: it shows **how the project works**, not how
it scored. Results charts belong in the story; the card gets the method.

- No axes, no grid, no tick chrome. Flat shapes on `SURFACE`; the figure's
  facecolor IS the background, so the preview fills edge-to-edge.
- Direct-labeled throughout; real names, dates, and protocol values from
  the frozen artifacts are welcome; **no metrics on the card**.
- Reference implementations (copy their structure): `method_schematic.py`
  (band layout: staircase/flow band + secondary band),
  `databricks_cycle_schematic.py` (loop/flow with a fail-closed branch),
  `gcp_flow_schematic.py` (pipeline with wine failure sites pinned at their
  nodes). Wine dashed annotations marking failure sites on a flow are house
  style — that honesty is the brand.
- Write the script in the project repo (generation lives where the data
  lives), render, and copy the PNG to
  `~/portfolio-site/assets/<projectid>-<slug>.png`.

## Deliverable 2 — story figures

- Generated from the project's frozen artifacts only, one per chapter that
  earns one. 2× retina (~2200 px wide for charts). Captions carry the
  numbers, the reading ("lower is better"), and sample sizes.
- Reference: `~/retailPRED-satellite/scripts/research_figures.py`
  (`--theme fieldnotes --scale 2`) for the themed chart pattern.
- Same deploy: `~/portfolio-site/assets/<projectid>-<slug>.png`.

## Deliverable 3 — the entry block and its story

Follow the sibling prompt's schema exactly. The narrative shape:

- **Prologue**: kicker `the question · where this started`, title `what was
  I investigating?` (or the project's equivalent), one paragraph: state the
  question plainly, then the surprising obstacle — and stop there.
- **A predecessor earns chapter 00.** If the project replaces or audits an
  earlier version, give it `chapter 00 · v1` with the old system's summary,
  its measured numbers, and the verdict — and end that section with the
  transition paragraph that forks the question into the chapters that
  follow ("…every chapter after this one is the same question asked one
  layer deeper").
- **Chapters**: kicker `chapter NN · <date or phase>`; one idea per
  paragraph, 3–7 sentences; figures placed where they belong; each chapter
  carries or links its evidence.
- **`result`**: the open path and the lesson — never a victory lap.
- **Voice**: first person, field-notes register. Measured numbers inline
  with their artifact named. Negative results as headline. No superlatives,
  nothing "planned" presented as done. Every sentence with a number must
  survive the check: which frozen file did this come from?
- If the project has a retired legacy stack, it gets its own first
  `technologyGroups` entry with `v1: true` — the renderer draws the wine
  **v1 sticker** and dashed chips, so retired skills read as history, not
  current claims. Never mix retired skills into current groups.
- Extend `techMark()` in app.js for any new glyph-worthy skill; unknown
  skills fall back to `⌘` — never leave a broken mapping for a skill that
  deserves one.

## Verification — run all, in order

1. `node --check projects.js && node --check app.js`.
2. **Cover-crop simulation** on the preview PNG (PIL): scale to the art box
   aspect with the cover rule (scale = max(box_w/w, box_h/h), center-crop)
   and confirm nothing legible is cropped — the whole title and caption
   line must survive.
3. Serve `~/portfolio-site` (`python3 -m http.server`), screenshot the
   gallery card, then open the dialog (click the card — an iframe wrapper
   with `contentDocument.querySelector('.project-card').click()` works in
   headless Chrome) and screenshot the summary (skill groups, sticker) and
   the story (chapter 00, figures). Fix collisions before shipping.
4. Read the entry aloud against the artifacts: every number → named file.
   Anything you cannot source, delete.

## Boundaries

- Scope: `projects.js`, `assets/`, and `app.js` **only if** a new glyph is
  needed. Never restyle the site, never touch other entries, no build step,
  no new dependencies — the site is plain static JS.
- Nothing fabricated: no invented numbers, no stock images, no skills not
  in the repos, no scope beyond the entry.
- Commit in `~/portfolio-site` with the Claude Code attribution line.
  **Pushing deploys the live site** — push on the owner's instruction.
- The `retailpred` entry and its scripts
  (`~/retailPRED-satellite/scripts/*.py`) are the reference
  implementations; when in doubt, imitate them rather than inventing.
