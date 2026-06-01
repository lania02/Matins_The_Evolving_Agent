# Matins

> A daily human–AI brainstorm loop. Each morning it proposes four research ideas,
> learns your taste from how you re-rank and comment on them, and consolidates
> durable lessons into a versioned "taste skill."

Matins is a small, **model-agnostic** adaptive-governance loop you run yourself.
The product is not the daily ideas alone — it is the **append-only log of
`(idea, self-rank, your-rank, comment)` tuples** that compounds over months and
makes both you and the system better. See [`DESIGN.md`](DESIGN.md) for the full
engineering rationale.

*(Project codename was "Mendel"; renamed to **Matins** — the pre-dawn canonical hour.)*

---

## How it works (one screen)

```
08:00  matins run         generate 4 ideas (4 slots) → novelty check → push digest
~11:00 matins collect     pull your replies → store feedback → reflect (rank divergence)
weekly matins consolidate propose a taste-skill update → you approve → version it
```

- **Four slots** per day (DESIGN §6): A high-fit (exploit), B adjacent-stretch,
  C orthogonal (contrarian), D random-mutation. Variation is built in so the
  system never collapses onto what you already like.
- **Anti-repetition guard.** Recently proposed idea titles — *and the ideas already
  generated earlier in the same batch* — are fed back into the prompts with a hard
  "must be distinct" constraint, so the slots never re-surface near-duplicates of last
  week's ideas nor collapse onto each other (the adjacent slot can't restate high-fit).
  Each slot also retries (resampling the random slot's genes) so you reliably get all
  four ideas, not three.
- **Anti-red-ocean gate (optional).** A contrarian "jump to a distant domain" tends to land
  on that domain's most *famous* topic — a saturated red ocean (e.g. GNNs for protein folding)
  that inspires no one. When enabled, a gated slot's candidate is grounded in **real
  literature density** (OpenAlex work-count, binned into a calibrated band) plus its closest
  existing works, and a judge regenerates it only if the area is *both* saturated *and* the
  idea is an undifferentiated textbook pairing — a genuinely novel angle inside a busy field
  still passes. **Which slots to gate is yours to choose and depends on your research
  direction** (`novelty.saturation_gate_slots`, default `[orthogonal]`): the orthogonal slot
  is the one prone to this, while gating high-fit (which *should* exploit your
  possibly-crowded core) or the random probe (deliberately unbiased) usually does more harm
  than good. An empty list turns it off; it is inactive offline (no search provider).
- **The log is the asset.** The two memory tiers (fast / slow) are *computed* from
  the log by convolving temporal kernels (DESIGN §5), never separately maintained.
- **Human in the loop on learning, not just generation.** A skill edit requires
  your approval and persistence across observations — a tired morning never
  rewrites your taste.
- **Adaptive learning layer** (see [`algo-upgrade-plan.md`](algo-upgrade-plan.md)).
  Residuals are first-class (the random slot is treated as a clean, exploit-free taste
  probe; ideas you rank far above the system are surfaced as "positive surprises"),
  comments are routed by kind (taste / novelty / feasibility / structure), the explore
  temperature adapts to how volatile your recent agreement has been, a quality-diversity
  archive can revive dormant-but-liked directions, and an optional, human-approved
  **self-evolution** step proposes brand-new taste dimensions — each gated by a held-out
  backtest before it can be adopted (`consolidation.evolve_dimensions`, off by default).

---

## Install

Requires Python 3.10+.

```bash
pip install -e .          # installs the `matins` command
cp config.example.yaml config.yaml
```

Dependencies are deliberately minimal: `httpx` + `PyYAML` + the standard library.

## Configure

Edit `config.yaml` (see [`config.example.yaml`](config.example.yaml) and DESIGN §13):

- **Provider** — defaults to Anthropic (`claude-opus-4-8`). Set your key:
  `export MATINS_API_KEY=...`. Swap to `openai` or `openai_compatible`
  (Ollama / vLLM / LM Studio) by changing `provider.name` + `base_url`.
- **Idea language** — `output_language: bilingual` (English term + Chinese gloss),
  or `en` / `zh`.
- **Messaging** — `telegram` by default. Create a bot via BotFather, then:
  `export MATINS_TELEGRAM_TOKEN=...` and run `matins init-telegram` to discover
  your `chat_id`. Set `channel: none` to use the CLI only.
- **Interests** — fill in [`prompts/interest_seed.md`](prompts/interest_seed.md)
  (your standing interests, always fed) and `retrieval.sources` (keyword queries
  for the daily fresh-literature feed) so ideas are actually about your field.
- **Fresh-literature feed** — each morning the generator blends a small, balanced
  set of fresh items across sources, tagged by origin in the prompt:
  - `openalex` (scholarly, cross-domain, citation-aware) and `arxiv` (fresh
    preprints) — the scholarly backbone;
  - `tavily` (web / why-now, reuses `TAVILY_API_KEY`) and `hackernews` (community
    signal) — a minority of timeliness / breakout.

  The mix is set by `retrieval.blend` (per-source quotas), interleaved and capped at
  `retrieval.max_items` — a deliberate blend, not a flat pile. Set a source to `0` to
  drop it; `openalex`'s key (`OPENALEX_API_KEY`) is optional (lifts the rate limit),
  `arxiv`/`hackernews` are keyless, `tavily` is skipped without its key.

## Daily use

```bash
matins run                 # generate + send today's four ideas
matins collect             # a few hours later: ingest your reply
matins consolidate         # weekly: propose a taste-skill update
matins consolidate --approve 3   # accept proposed skill version 3
matins feedback "3>1>4>2"  # offline fallback when not using Telegram
matins dig 3               # on-demand deep dive: a grounded, cited briefing for idea #3
matins favorites           # list ideas you flagged "must try" (mirrored to favorites.md)
```

`matins run` is **idempotent per date** — re-running on the same day returns the
stored batch verbatim (no regeneration, no new fetch). Only a new date triggers a
fresh batch.

**Reply format** (DESIGN §9.4) — rank best→worst by idea number, optional comments.
Two inline commands are recognized in a reply and acted on during `matins collect`:

```
3>1>4>2
#3 explicit isomorphism, tractable first step
#1 already done by <author> 2024
must try #3                # copy idea #3 into your favorites library
dig #1                     # request a deep-dive briefing for idea #1
```

> **Note on `dig` via Telegram reply:** the reliable path is the CLI `matins dig N`
> (add `--send` to also push the brief to your channel). The "dig #N" *reply* path
> depends on Telegram's single-consumption update offset, which is shared between a
> scheduled `collect` and a manual one — a reply consumed by one run is invisible to
> the other. Prefer the CLI when you want a deep dive to definitely happen.

## Scheduling (no daemon)

All commands are idempotent and meant to be triggered externally.

**Linux/macOS cron:**
```
0  8 * * *   cd /path/to/matins && matins run        >> data/cron.log 2>&1
0 11 * * *   cd /path/to/matins && matins collect     >> data/cron.log 2>&1
0  9 * * 1   cd /path/to/matins && matins consolidate  >> data/cron.log 2>&1
```

**Windows Task Scheduler:** schedule the same three commands.

## Testing sandbox

To experiment with generation / `dig` / deep dives **without touching your real
log**, run against a throwaway database. Set `state_dir` in a separate config and
all *mutable* state (db, favorites, deep-dive mirrors) is redirected under that
folder, while `prompts/`, `skills/`, and your interest seed stay shared:

```bash
cp sandbox.config.example.yaml sandbox.config.yaml   # then set provider to match config.yaml
matins --config sandbox.config.yaml run --date sbx-1
matins --config sandbox.config.yaml run --date sbx-2   # exercises the anti-repetition guard vs sbx-1
matins --config sandbox.config.yaml dig 1
```

The example ships with `state_dir: sandbox` and `messaging.channel: none`, so the
sandbox writes only to `./sandbox/` and never calls Telegram or disturbs the real
update offset. Reset any time by deleting `./sandbox/`. (`sandbox/` and
`sandbox.config.yaml` are gitignored.) `python -m matins ...` works as an alias for
the `matins` command when running an in-tree checkout.

## Layout

```
matins/
  config.py                 typed config
  store/      db.py models.py        append-only SQLite log + derived queries
  providers/  base.py anthropic.py openai.py openai_compatible.py search_web.py
              messaging/ base.py telegram.py whatsapp_*.py
  generate/   pipeline.py slots.py schema.py novelty.py saturation.py deepdive.py explore.py
  memory/     kernels.py consolidate.py backtest.py evolve.py
  feedback/   capture.py diverge.py
  digest/     render.py
  cli.py  __main__.py
prompts/      slot_*.txt self_rank.txt predict_rank.txt summarize_recent.txt
              propose_skill_diff.txt propose_dimension.txt saturation_judge.txt
              deepdive_*.txt genes.yaml interest_seed.md
skills/       taste.md     human-readable mirror of the active skill version
algo-upgrade-plan.md        the adaptive / self-evolution upgrade plan (Phases 1-5)
data/         matins.db    (gitignored)
deep_dives/   <slug>.md    deep-dive briefings (gitignored)
favorites.md  curated "must try" ideas (gitignored)
sandbox.config.example.yaml   template for an isolated testing sandbox
tests/
```

## Honest expectations

The first weeks will be mediocre (cold start). This is an instrument measured in
**months, not days** — it optimizes for data quality and recomputability, because
that is the only thing that compounds. See DESIGN §15 for risks (Goodhart,
skill rot, reflexive taste drift) and their mitigations.

## License

MIT — see [`LICENSE`](LICENSE).
