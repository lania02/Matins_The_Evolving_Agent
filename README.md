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
- **The log is the asset.** The two memory tiers (fast / slow) are *computed* from
  the log by convolving temporal kernels (DESIGN §5), never separately maintained.
- **Human in the loop on learning, not just generation.** A skill edit requires
  your approval and persistence across observations — a tired morning never
  rewrites your taste.

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
  and `retrieval.sources` so day-one ideas are actually about your field. The
  repo ships this as a template; **it starts empty.**

## Daily use

```bash
matins run                 # generate + send today's four ideas
matins collect             # a few hours later: ingest your reply
matins consolidate         # weekly: propose a taste-skill update
matins consolidate --approve 3   # accept proposed skill version 3
matins feedback "3>1>4>2"  # offline fallback when not using Telegram
```

**Reply format** (DESIGN §9.4) — rank best→worst by idea number, optional comments:

```
3>1>4>2
#3 explicit isomorphism, tractable first step
#1 already done by <author> 2024
```

## Scheduling (no daemon)

All commands are idempotent and meant to be triggered externally.

**Linux/macOS cron:**
```
0  8 * * *   cd /path/to/matins && matins run        >> data/cron.log 2>&1
0 11 * * *   cd /path/to/matins && matins collect     >> data/cron.log 2>&1
0  9 * * 1   cd /path/to/matins && matins consolidate  >> data/cron.log 2>&1
```

**Windows Task Scheduler:** schedule the same three commands.

## Layout

```
matins/
  config.py                 typed config
  store/      db.py models.py        append-only SQLite log + derived queries
  providers/  base.py anthropic.py openai.py openai_compatible.py search_web.py
              messaging/ base.py telegram.py whatsapp_*.py
  generate/   pipeline.py slots.py schema.py novelty.py
  memory/     kernels.py consolidate.py
  feedback/   capture.py diverge.py
  digest/     render.py
  cli.py
prompts/      slot_*.txt self_rank.txt summarize_recent.txt propose_skill_diff.txt
              genes.yaml interest_seed.md
skills/       taste.md     human-readable mirror of the active skill version
data/         matins.db    (gitignored)
tests/
```

## Honest expectations

The first weeks will be mediocre (cold start). This is an instrument measured in
**months, not days** — it optimizes for data quality and recomputability, because
that is the only thing that compounds. See DESIGN §15 for risks (Goodhart,
skill rot, reflexive taste drift) and their mitigations.

## License

MIT — see [`LICENSE`](LICENSE).
