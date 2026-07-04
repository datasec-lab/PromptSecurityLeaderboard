# PromptSecurity Leaderboard

Static leaderboard for completed PromptSecurity experiments.

Live site:

`https://datasec-lab.github.io/PromptSecurityLeaderboard/`

## Make a Contribution

The public page includes a contribution section with links to:

- PromptSecurity code and evaluation data: `https://github.com/datasec-lab/PromptSecurity`
- Issue submissions: `https://github.com/datasec-lab/PromptSecurity/issues`
- Pull requests: `https://github.com/datasec-lab/PromptSecurity/pulls`
- This leaderboard site: `https://github.com/datasec-lab/PromptSecurityLeaderboard`

## 1) Build data

From repo root:

```bash
python leaderboard_site/scripts/build_leaderboard_data.py \
  --input-dir experiments/placeholders \
  --utility-input-dir experiments/placeholders_utility \
  --output leaderboard_site/data/leaderboard.json
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

The workflow rebuilds `leaderboard_site/`, bundles only paper-compatible run payloads into `leaderboard_site/data/runs/` for the comparison popup, and publishes the static site to the target branch.

## Data assumptions

- Reads paper-compatible placeholders with primary GPT-bin scores; status is used only to choose the best duplicate for the same paper key.
- Schema `v3` includes a paper-compatible `paper` block for default site tables.
- The default generated JSON is aggregate-only: `meta`, `overview`, and `paper`.
- Public publishing uses `--bundle-runs-dir` to include only paper-compatible run-level payloads needed by the comparison popup.
- The default leaderboard is restricted to the paper main experiment:
  - 11 configured target models.
  - 20 configured attack settings.
  - 9 configured defenses, with `no_defense` used only as a matched baseline.
  - HarmBench-style GPT-4.1-mini judger (`gpt_judger_harmful_binary`) as the primary harmful-output judger.
- Auxiliary runs such as `no_attack` baselines are retained only for derived metrics; ABJ assistant-LLM ablations and other non-main attack variants are excluded from leaderboard rankings.
- Leaderboard ASR uses only `gpt_judger_harmful_binary`.
- HarmBench, HarmBench-style GPT-4.1-mini, and prefix judger results are exposed as independent comparison views; they are not averaged together.
- `ASR` is derived from sample-level judger outputs:
  - `0` means safe.
  - `1` means unsafe.
- Model rankings use no-defense ASR and matched clean-baseline transition metrics.
- Defense rankings use matched no-defense counterfactuals on shared sample identifiers.
- Attack and defense rankings are split by black-box-compatible and white-box-only method access.
- Utility deltas are read from `experiments/placeholders_utility` when available.
- Headline matrices do not fill missing no-defense cells from defended runs.
