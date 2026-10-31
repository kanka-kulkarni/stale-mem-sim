# Re-checking stale user memory: simulator and experiments

Code for the paper *"[title withheld for review]"*, submitted to the NAACL 2027
Industry Track.

An assistant stores facts about a user. The facts change silently between sessions. Each
session brings a task, and the assistant must decide which stored facts to re-ask before
answering. This repository contains:

- a deterministic simulator in which the correct answer is known exactly;
- the re-ask policies compared in the paper (TTL, volatility, staleness x impact, budget,
  information gain, learned per-user rates, an oracle, and per-session LLM policies);
- matched-cost evaluation with paired exact McNemar tests and Holm correction;
- one YAML config per study, covering every result reported in the paper;
- the LLM rating tables and every cached LLM reply, so the LLM-dependent results can be
  reproduced exactly.

The core simulator uses no language model. LLMs enter only as optional policies, raters and
answerers, through a local [Ollama](https://ollama.com) server. All their replies are cached
in `calibration/llm_cache.jsonl`, so the LLM results can be replayed without a GPU or Ollama.

## Setup

Requires Python 3.10 or newer. Everything runs on a CPU-only laptop.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests (163 tests, about 20 seconds):

```bash
python -m unittest discover -s tests -t .
```

## Quick start

Every experiment runs through one entry point, with its settings in a YAML config:

```bash
python evaluate.py --config configs/quickstart.yaml
```

This runs four policies on 10 simulated users x 20 sessions of meal planning in under a
second. It prints each policy's accuracy and questions per session, checks three sanity
conditions (`always` is 100% accurate, `never` declines over sessions, and every policy
sees identical sessions), and writes per-session records to `results/quickstart/`.

```bash
python evaluate.py --config configs/main.yaml --list                 # the config's stages
python evaluate.py --config configs/main.yaml --stages meals         # one stage
python evaluate.py --config configs/main.yaml --dry-run              # print the commands only
```

Each stage runs one module in `experiments/`, and its YAML arguments map to that module's
command-line flags (`llm_impact: x.csv` becomes `--llm-impact x.csv`), so `--dry-run` shows
exactly what will run. To try other settings, copy a config and edit it.

Policy specs: `never`, `always`, `always_all`, `oracle`, `ttl(N)`, `volatility(K)` (add
`,blind` to either for the task-blind version), `info_gain(threshold)`,
`sxi(threshold,impact)` (staleness x impact) and `budget(n,impact)`, where impact is
`exact`, `coarse` (the dependency list) or `llm` (with `llm_impact: <table>`). See
`make_policy` in `simulator/policies.py`.

## Repository layout

```
simulator/       core model, domains, policies, matching, metrics and statistics
  core.py          attributes, life events, tasks, paired timelines, memory, session loop
  domains/         commute.py and meals.py (facts, tasks, decision rules, life events)
  policies.py      re-ask policies, all with decide(memory, task, t) -> facts to ask
  matching.py      tuning a policy to a target question cost on tuning seeds
  stats.py         exact McNemar test and Holm correction
  theory.py        selection rules and bounds of Proposition 1
  llm.py           cached Ollama client (temperature 0, seed 0, JSON output)
  impact_llm.py, volatility_llm.py, llm_policy.py   LLM rater and policy prompts
evaluate.py      entry point: runs the stages of a config
configs/         one YAML config per study (see below)
experiments/     one module per experiment, called by evaluate.py
calibration/     LLM rating tables, the LLM reply cache and the dataset calibration scripts
results/         experiment outputs (created when you run a config)
tests/           unit tests
```

## Design in brief

- **Users.** Each fact has a value set and a volatility class with a per-session change
  probability (static 0, slow 0.03, fast 0.15; meal-planning rates are calibrated on
  VitaBench 2.0). A fact that changes moves to a uniformly random other value. Meal
  planning adds life events that change several facts together.
- **Sessions.** At onboarding (session 0) memory equals the true state. Each of 20
  sessions brings a task drawn uniformly. The policy picks facts to re-ask, the simulated
  user answers truthfully, the answer is computed from memory by the task's decision rule,
  and it is scored against the answer from the true state.
- **Paired design.** Each user's timeline (true states and tasks) is generated once from
  `(seed, user)` and replayed by every policy. Drift never depends on what a policy asked.
- **Matched cost.** Each policy's parameter is tuned on tuning seeds 1000-1004 to 25%, 50%
  or 75% of `always`'s questions, mixing two neighbouring settings to hit the target. All
  reported numbers come from held-out seeds 0-4 (50 users each, so 250 users x 20
  sessions).

## What is included

The repository contains code and inputs, not results. The rule-based experiments are
deterministic given the seeds, so running a config reproduces the paper's numbers.

Two kinds of LLM output are included because they cannot be regenerated exactly: 4-bit
models at temperature 0 can still answer differently on another Ollama version or CPU.

- `calibration/llm_impact_*.csv` and `calibration/llm_volatility_*.csv` hold each model's
  relevance and change-rate ratings. The rule-based experiments read them.
- `calibration/llm_cache.jsonl` holds every LLM reply behind the per-session and end-to-end
  results.

## Reproducing the paper

Run the configs in this order. Later configs read the outputs of earlier ones.

| Order | Config | Paper results | Runtime |
|---|---|---|---|
| 1 | `configs/main.yaml` | Matched-cost comparison (Table 1, appendix curves and tables), both domains, with the oracle, Always-all and information-gain baselines | CPU, no model calls |
| 2 | `configs/per_session_llm.yaml` | An LLM chooses re-checks each session (Section 5.3, Appendix G) | replayed from the cache |
| 3 | `configs/end_to_end.yaml` | An LLM writes the answer (Section 5.4, Appendix H) | replayed from the cache |
| 4 | `configs/value_study.yaml` | Same selection, different value estimates (Section 5.2, Appendix E); LLM raters (Table 3, Appendix F) | CPU, no model calls; E1-E2 about 11 min |
| 5 | `configs/robustness.yaml` | Per-user rates, wrong answers and life-event strength (Section 5.5, Appendix I) | CPU, about 4 min per analysis |
| 6 | `configs/calibration.yaml` | VitaBench change rates and the Instacart check (Appendices C and J); needs the downloaded datasets | minutes |

For example:

```bash
python evaluate.py --config configs/main.yaml
```

Each study writes its summaries, matched-cost settings and paired tests (exact McNemar
with Holm correction) as CSV and JSON files under `results/`. The paper's figures and
tables are drawn from these files; the plotting scripts are not part of this repository.

The LLM configs replay replies from `calibration/llm_cache.jsonl`, so they need Ollama only
for prompts that are not cached. The answerer's repeat-agreement check (`end_to_end`, stage
`determinism`) always calls the model. `configs/llm_ratings.yaml` regenerates the rating
tables from the cache. Two tests skip until the files they check exist; they say which
config to run.

The decision-rule diagnostic checks that every value of a fact can change some answer. Run
it with `python -m experiments.impact_table --domain commute` (or `meals`).

### LLMs

All models run through Ollama at temperature 0 and seed 0 with JSON output, using the
default 4-bit (Q4_K_M) tags: `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b`, `llama3.1:8b`,
`qwen3.5:9b`, `gemma4:e4b`, `phi4-mini:3.8b`, `granite4.1:8b` and `qwen3:8b`. Thinking is
turned off (`no_think: true`) for Qwen3.5, Gemma 4 and Qwen3. The prompts are in
`simulator/impact_llm.py`, `simulator/volatility_llm.py`, `simulator/llm_policy.py` and
`experiments/e2e.py`, and are reproduced verbatim in the paper's appendix.

### Real-data calibration

The datasets are not redistributed. Download them and set their paths in
`configs/calibration.yaml`.

- **VitaBench 2.0** (meal-planning change rates). Download `tasks.json` from the
  [dataset page](https://huggingface.co/datasets/meituan-longcat/VitaBench-2.0). Each
  subtask transition counts as one session, and a tag's rate is (updates + deletes) / tags
  held, with 95% Wilson intervals. The resulting rates are written into
  `simulator/domains/meals.py`.
- **Instacart** (a check for temporal structure in order histories, which found none). The
  data needs a Kaggle account and is licensed for non-commercial use only.

## Licence

Code: MIT (see `LICENSE`).
