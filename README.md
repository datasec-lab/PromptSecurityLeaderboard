# Leaderboard Site

Static leaderboard for completed PromptSecurityEval experiments.

## 1) Build data

From repo root:

```bash
python leaderboard_site/scripts/build_leaderboard_data.py \
  --input-dir experiments/placeholders \
  --output leaderboard_site/data/leaderboard.json
```

For static hosting (GitHub Pages), bundle completed run payload JSONs too:

```bash
python leaderboard_site/scripts/build_leaderboard_data.py \
  --input-dir experiments/placeholders \
  --output leaderboard_site/data/leaderboard.json \
  --bundle-runs-dir leaderboard_site/data/runs
```

## 2) Serve locally

From repo root:

```bash
python -m http.server 8080
```

Open:

`http://localhost:8080/leaderboard_site/`

## Live traffic counter (GoatCounter)

The page includes a live traffic row (Total / This Page / Last 30 Days).

Configure in `leaderboard_site/index.html`:

```html
<body data-goatcounter-code="your-goatcounter-code">
```

Examples:

- `data-goatcounter-code="promptsecurityeval"`
- `data-goatcounter-code="https://promptsecurityeval.goatcounter.com"`

Notes:

- The site auto-loads GoatCounter `count.js` and refreshes counters every 30s.
- If counters fail, check GoatCounter site settings for visitor count visibility/API access.

## Private main repo -> public leaderboard repo (one-click)

Workflow file:

`/.github/workflows/deploy_leaderboard_pages.yml`

This workflow builds from the private main repo and publishes the static site to a separate public repo branch.

### A) Prepare target public repo

1. Create a public repo (example: `your-org/promptsecurityeval-leaderboard`).
2. In that public repo, enable GitHub Pages:
   - Settings -> Pages
   - Source: `Deploy from a branch`
   - Branch: `gh-pages` (or the branch you choose), folder: `/ (root)`

### B) Configure source private repo (this repo)

In Settings -> Secrets and variables -> Actions:

1. Add repository variable:
   - `LEADERBOARD_PUBLIC_REPO` = `owner/repo` of the public target repo.
2. Optional repository variable:
   - `LEADERBOARD_PUBLIC_BRANCH` = publish branch (default `gh-pages`).
3. Add repository secret:
   - `LEADERBOARD_PUBLISH_TOKEN` = PAT that can push to the target public repo.

PAT recommendation:

- Fine-grained PAT with `Contents: Read and write` on the target public repo only.
- Classic PAT with `repo` scope also works, but is broader.

### C) Run publish

1. Push to `main` (auto trigger), or
2. GitHub -> Actions -> `Publish Leaderboard To Public Repo` -> `Run workflow`.

The workflow rebuilds `leaderboard_site/` and force-publishes it as an orphan commit to the target branch.

## Data assumptions

- Reads only completed-like runs (`success`, `completed`, `complete`).
- `ASR` is derived from sample-level judger outputs:
  - `0` means safe.
  - `1` means unsafe.
  - multi-judger dict/list values are averaged to `[0,1]`.
- Matrix prefers `no_defense` runs; if missing, it falls back to all-defense averages.
- With `--bundle-runs-dir`, run payload files are copied and run paths are rewritten to `data/runs/*.json`.
- Publishing this site means run payload data is publicly accessible in the target repo Pages site.
