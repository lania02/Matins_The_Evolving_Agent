# Matins — Engineering Design (v1)

> A daily human–AI brainstorm loop. Each morning it proposes four research ideas,
> learns your taste from how you re-rank and comment on them, and consolidates
> durable lessons into a versioned "taste skill" — built as a small, model-agnostic
> adaptive-governance loop you can run yourself and push to GitHub.

*`Matins` (the pre-dawn canonical hour — fitting a tool that proposes ideas each morning) was
chosen over the original codename `Mendel`. The mechanism is still variation + selection by a
human breeder.*

---

## 1. Purpose & non-goals

**Purpose.** Sustain creative idea-generation over months without (a) the human
doing the cold-start work every day, and (b) the system collapsing onto what the
human already likes. The product is not the daily ideas alone — it is the **log of
(idea, self-rank, your-rank, comment) tuples** that accumulates and makes both the
human and the system better over time.

**Explicit non-goals for v1.**
- No reward-model training, no fine-tuning. With one user and ~4 ideas/day, the data
  is far too sparse for gradient-based preference learning. Taste is learned
  **in-context** via a versioned skill, not by training weights.
- No always-on daemon, no heavy UI.
- "Internalize *structure* not *topic*" is **experimental and OFF by default** (see §10).
  It needs far more data before any conclusion is warranted. v1's job is to make that
  data cheap to accumulate, not to draw the conclusion.

**Honest expectations.** The first weeks will be mediocre (cold start). This is an
instrument measured in months, not days. The design optimizes for *data quality and
recomputability*, because that is the only thing that compounds.

---

## 2. Design philosophy

Five principles, each tied to a known failure mode.

1. **Model-agnostic.** All LLM, search, and messaging calls go through thin provider
   interfaces. Claude is a good default, but anyone with an OpenAI-compatible or local
   endpoint can reproduce the framework. No Claude-only features in the core loop.

2. **Variation–selection, not optimization-to-fit.** Each day emits four ideas across
   four *slots* (§6). Pure fit-maximization would converge to what you already like and
   kill novelty — the same lock-in dynamic that motivates the research this tool serves.
   Variation is built into generation; the human ranking is the selection pressure.

3. **The log is the asset; memory is a derived view.** An append-only local database
   stores raw material. The "two-tier memory" is **not** a separately maintained state —
   it is *computed from the log by reading windows of different lengths and aggregating each
   with an LLM* (§5). Consequence: you never commit to a consolidation rule now; you log
   everything and treat consolidation as a tunable transform you can revise or backtest later.

4. **Rankings are noisy measurements.** A tired morning is not a preference. Nothing is
   internalized from a single observation; consolidation requires *persistence* and your
   approval (§8). Same signal-vs-noise discipline that applies to the research itself.

5. **Reflexivity is expected (Lucas critique).** Your taste will drift *because* the system
   feeds you ideas. A skill fit to past rankings will mis-predict future taste. v1 does not
   "solve" this; it (a) keeps an explore slot so the system never assumes a static target,
   and (b) logs enough to study the drift later.

---

## 3. System overview

```
        ┌─────────────── daily trigger (cron / Task Scheduler / manual) ───────────────┐
        │                                                                               │
        ▼                                                                               │
  ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────────────────┐  │
  │ Context assembly  │──▶│ Generation (4 slots)  │──▶│ Novelty check (per idea)     │  │
  │ - taste skill     │   │  A high-fit           │   │ closest prior art via search │  │
  │ - fast memory     │   │  B adjacent-stretch   │   └──────────────────────────────┘  │
  │ - fresh retrieval │   │  C orthogonal         │                                      │
  │ - interest seed   │   │  D random mutation    │            ┌──────────────────────┐  │
  └───────────────────┘   │  + self-rank+rationale│──────────▶│ Digest render        │  │
        ▲                 └──────────────────────┘            └──────────┬───────────┘  │
        │                                                                ▼               │
  ┌───────────────────┐                                    ┌──────────────────────────┐ │
  │ Slow memory /      │                                    │ Messaging channel (§9)   │ │
  │ skill (versioned)  │                                    │  default: Telegram       │─┼──▶ you
  └─────────┬─────────┘                                     │  optional: WhatsApp      │ │
        ▲   │                                               └──────────┬───────────────┘ │
        │   │                                                          │  your reply      │
  ┌─────┴───┴──────────┐   ┌──────────────────────┐                    ▼  (rank+comment)  │
  │ consolidation      │◀──│ Reflection           │◀── fetch_replies ──┘                  │
  │ (weekly, approved) │   │ - rank divergence    │                                       │
  └────────────────────┘   │ - hypothesis update  │     ALL events ─▶ append-only LOG ────┘
                           └──────────────────────┘                    (SQLite, local)
```

Everything writes to the **log**; the two memory tiers are *reads* of that log through
kernels (§5). The messaging channel is the only thing the human touches day to day.

---

## 4. Data model (the asset)

SQLite, single file, no server — keeps the framework dependency-light and portable.
Append-only in spirit: rows are inserted, not destructively overwritten. The skill
(slow memory) is versioned rather than mutated in place.

```sql
CREATE TABLE batches (
  batch_id        TEXT PRIMARY KEY,      -- uuid
  date            TEXT,
  skill_version   INTEGER,
  temperature     REAL,                  -- explore aggressiveness knob (0..1)
  provider        TEXT, model TEXT,
  self_user_tau   REAL,                  -- Kendall tau between self-rank and user-rank
  digest_msg_id   TEXT,                  -- messaging message id of the sent digest
  created_at      TEXT
);

CREATE TABLE ideas (
  idea_id         TEXT PRIMARY KEY,
  batch_id        TEXT REFERENCES batches(batch_id),
  slot            TEXT,                  -- 'highfit' | 'adjacent' | 'orthogonal' | 'random'
  idx             INTEGER,               -- 1..4, the number shown to the user
  title           TEXT,
  mechanism       TEXT,
  why_now         TEXT,
  math_structure  TEXT,
  prior_art       TEXT,                  -- filled by novelty check (§7)
  tractability    TEXT,
  fit_to_program  TEXT,
  random_genes    TEXT,                  -- for slot=random: sampled (domain,method,constraint)
  self_rank       INTEGER,
  self_rationale  TEXT,
  created_at      TEXT
);

CREATE TABLE feedback (
  idea_id         TEXT REFERENCES ideas(idea_id),
  user_rank       INTEGER,               -- 1..4
  user_comment    TEXT,                  -- optional one-liner; richest signal we get
  source          TEXT,                  -- 'telegram' | 'cli' | 'card'
  created_at      TEXT
);

CREATE TABLE retrieval_log (
  batch_id TEXT, query TEXT, source TEXT,
  result_ids TEXT, created_at TEXT       -- result_ids used to dedup future batches
);

CREATE TABLE taste_hypotheses (           -- FAST memory
  hyp_id TEXT PRIMARY KEY,
  text TEXT,
  kind TEXT,                              -- 'topic' | 'structure'  (cheap tag, see §10)
  evidence TEXT,                          -- json list of idea_ids
  confidence REAL, occurrence INTEGER,
  status TEXT,                            -- 'open'|'confirmed'|'rejected'|'retired'
  first_seen TEXT, last_seen TEXT
);

CREATE TABLE skill_versions (             -- SLOW memory
  version INTEGER PRIMARY KEY,
  content TEXT, parent_version INTEGER,
  diff_summary TEXT, approved INTEGER,    -- requires human approval to activate
  created_at TEXT
);

CREATE TABLE messaging_state (            -- offset bookkeeping for reply polling
  channel TEXT PRIMARY KEY,
  last_update_id TEXT
);
```

Because every memory artifact is *derived* from `ideas`/`feedback`, you can drop and
recompute memory at any time, or re-derive it under a different kernel config without
losing anything. That is the whole point of logging raw material first.

---

## 5. Memory as windowed reads of the log

Treat the ordered feedback log as a time series of events `E = [e_1 ... e_t]` (each `e`
= one idea + its feedback). A **memory kernel** is a windowed aggregation over `E`:

```yaml
memory_kernels:
- name: fast
  window_days: 7        # look back 7 days
  stride: 1             # every batch, dense
  aggregator: llm_summarize_recent   # high-resolution snapshot of current preference
  feeds: generation                  # injected fresh each day
- name: slow
  window_days: 75       # long horizon
  stride: 1             # every batch (default); stride>1 = optional batch subsampling
  aggregator: llm_propose_skill_diff # consolidated, stable structure
  feeds: consolidation               # proposes skill edits (human-approved)
```

- **Fast memory** = short window, stride 1 → dense, recent, drift-sensitive. Recomputed
  every morning and inlined into the generation prompt.
- **Slow memory** = long window → many days of feedback aggregated by the LLM into a
  proposed skill diff. Feeds consolidation. (An optional `stride>1` subsamples batches to
  shrink the prompt; that is plain decimation, *not* a low-pass filter — the default is `1`,
  i.e. keep the whole window. Genuine denoising is the consolidation recurrence threshold +
  human approval, §8, not the sampling step.)

Properties this buys us:
- **Recomputable & reconfigurable.** Window/stride/aggregator are config, not code.
- **Backtestable.** Hold out the last *k* batches, compute memory from the rest, and check
  how well each kernel config predicts your held-out rankings. This is how you'll
  *eventually* decide things like "does internalizing structure help?" (§10) — empirically,
  on accumulated data, not by guessing now.
- **No destructive state.** Memory is a view; the log is the truth.

---

## 6. Generation: four slots

### 6.1 The slots

| Slot | Conditioning | Role |
|------|--------------|------|
| **A — high-fit** | strong: taste skill + fast memory | exploit; "you'll probably like this" |
| **B — adjacent-stretch** | medium: high-fit core, one dimension pushed out of comfort | controlled exploration |
| **C — orthogonal** | inverted: deliberately violate a learned preference / distant domain | contrarian; tests the boundary |
| **D — random mutation** | minimal: forced combination of randomly sampled "genes" | pure perturbation; anti-overfitting |

A single **temperature** knob (0..1) scales how far B, C, D depart from A and maps to the
API sampling temperature for those slots. Logged per batch so its effect is measurable.

### 6.2 Idea schema

Every idea is generated as structured JSON (tolerant-parsed, §11) with: `title`,
`mechanism`, `why_now`, `math_structure` (empty if none — itself a signal), `tractability`,
`fit_to_program`, and `prior_art` (blank at generation, filled by §7).

### 6.3 The random-mutation gene pool

Slot D samples a triple `(domain, method, constraint)` from a vocabulary and asks the model
to force a coherent idea out of it. v1 seeds the vocabulary from `prompts/genes.yaml`; later
it can be **grown from the log**, the embryo of the v2 evolutionary engine (§16).

---

## 7. Novelty check

For each idea, search for the closest existing work and attach a one-line verdict + link to
`prior_art`. Operationalizes the rule that *novelty is an empirical claim about the
literature, not derivable from how elegant the idea feels*.

- Goes through a pluggable `SearchProvider` (web and/or arXiv). If none is configured, the
  step is skipped and the idea is flagged `prior_art: "[unchecked]"`.
- Advisory, not a filter: a high-overlap hit is shown to you, not auto-deleted.

---

## 8. Feedback, reflection, consolidation

**Feedback capture.** The default channel is **Telegram** (§9): the digest is pushed to you,
you reply with a ranking + optional comments, and `matins collect` ingests them. A CLI mode
and an editable-markdown-card mode are kept as offline fallbacks (`source` is logged per
feedback row). Both ordinal rank (1–4) and a free-text one-liner are captured; the comment is
the densest signal we get and is never required but always invited.

**Reflection.** After feedback, compute Kendall's tau between self-rank and your rank; store
on the batch. When divergence is high, the system writes a short diagnosis — which feature it
mis-weighted — as a new or reinforced `taste_hypothesis` (occurrence++, confidence updated).
Single-batch divergence never edits the skill.

**Consolidation.** Triggered weekly or when a hypothesis crosses an occurrence threshold. The
slow kernel produces a **proposed skill diff**, pushed to you over the same channel; you
approve by reply; on approval a new `skill_version` is committed (versioned → rollback trivial).
The human is in the loop on the *learning* step, not only generation.

---

## 9. Messaging & interaction channel

### 9.1 Why Telegram is the default and WhatsApp is optional

Binding WhatsApp is possible, two ways: **(1) unofficial** — the Baileys library drives the
WhatsApp Web protocol via QR pairing to a (ideally dedicated) number; simple, no Meta
approval, but WhatsApp's terms prohibit automation (ban risk is low at personal volume but
nonzero) and it is fragile (it mirrors your phone — phone offline = bot offline). **(2)
official** — Meta's WhatsApp Business Cloud API: dedicated business number, access token,
webhooks, an HTTPS-reachable server; compliant but heavier. Extra friction for *this* project:
Baileys is Node/TS, so the unofficial route needs a Node side-process or a Go-based library
(whatsmeow / neonize) — a cross-language bridge. Telegram has first-class Python support and a
sanctioned Bot API.

**Decision:** ship Telegram as the default channel; expose a channel-agnostic
`MessagingProvider` so a WhatsApp adapter can be added if you accept the trade-offs.

### 9.2 MessagingProvider interface

```python
# providers/messaging/base.py
class Reply(TypedDict):
    text: str
    ts: str
    reply_to_message_id: str | None
    update_id: str

class MessagingProvider(Protocol):
    def send(self, text: str, *, parse_mode: str = "MarkdownV2") -> str: ...     # -> message_id
    def fetch_replies(self, since_update_id: str | None) -> list[Reply]: ...
```

### 9.3 Telegram adapter (default)

- **Setup (once):** create a bot via BotFather → put the token in `MATINS_TELEGRAM_TOKEN`;
  message the bot once and read your `chat_id` (a `matins init-telegram` helper prints it).
- **Send:** Bot API `sendMessage`. The digest is one header message + four idea messages
  (`#1`–`#4`), each well under Telegram's 4096-char limit; `prior_art` shown in short form.
  `digest_msg_id` is stored on the batch.
- **Collect:** Bot API `getUpdates` with the stored `last_update_id` offset (in
  `messaging_state`). **No always-on daemon** — a second cron runs `matins collect` a few
  hours after `matins run`, pulls replies since the digest, parses, writes feedback, advances
  the offset.

### 9.4 Interaction protocol (reply format)

You reply to the bot with:

```
3>1>4>2
#3 explicit isomorphism, tractable first step
#1 already done by <author> 2024
```

- **Line 1** = ranking best→worst by idea number. **Comment lines** = `#<n> <text>`, optional.
- The parser maps numbers→ranks, attaches comments, writes feedback rows. Tolerant: a missing
  rank is flagged back to you; missing comments are fine.
- **Guided mode (v1.x):** the bot asks one idea at a time with inline buttons (1–4) — nicer on
  mobile but requires a long-running listener, so it is deferred.

### 9.5 Daily flow (two crons, no daemon)

1. `08:00  matins run`      → generate → store → push digest to Telegram.
2. `~11:00 matins collect`  → getUpdates → parse replies → store feedback → reflect (tau, hypotheses).
3. weekly `matins consolidate` → propose skill diff → bot sends it → you approve by reply → commit version.

### 9.6 WhatsApp adapter (experimental, opt-in)

Same `MessagingProvider` interface. Two implementation paths, both off by default:
`whatsapp_baileys` (a small Node Baileys bridge exposing send/fetch over a local socket; QR
pairing on first run) or `whatsapp_cloud` (Meta Business Cloud API with webhook). Carries the
ToS / ban / fragility caveats in §9.1; use a dedicated number.

---

## 10. Structure-internalization (experimental, OFF by default)

Hypothesis: internalize the *structural* preference ("prefers an explicit isomorphism",
"rewards contrarian framing", "penalizes ideas with no first concrete step") rather than the
*topic* ("likes phase transitions this month"), because topics drift (Lucas) while structure
is more stable.

v1 stance: **do not commit.** What v1 does instead is cheap: tag every hypothesis `topic` or
`structure` at creation (`taste_hypotheses.kind`), so that *later* you can backtest whether
structure-tagged hypotheses predict held-out rankings better. If they do, flip the flag on.
The decision is deferred to evidence by design.

---

## 11. Provider abstraction (model-agnostic core)

```python
# providers/base.py
class LLMProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float, json_schema: dict | None) -> str: ...

class SearchProvider(Protocol):
    def search(self, query: str, *, k: int = 5) -> list[dict]: ...   # [{title, url, snippet}]

# MessagingProvider lives in providers/messaging/base.py (see §9.2)
```

- LLM adapters shipped: `anthropic`, `openai`, `openai_compatible` (covers local servers such
  as Ollama / vLLM / LM Studio and any OpenAI-API-shaped endpoint). Selected via config.
- **Portable structured output.** The core asks for JSON in the prompt and uses *tolerant
  parsing*: strip code fences, parse, validate against the idea schema, one repair-retry on
  failure. Providers with native JSON modes can opt in via the adapter.
- Optional `litellm` adapter to cover many providers with one dependency, kept optional.

---

## 12. Repository layout

```
matins/
  README.md  DESIGN.md  LICENSE
  pyproject.toml            # stdlib + sqlite + httpx, minimal
  config.example.yaml
  prompts/
    slot_highfit.txt slot_adjacent.txt slot_orthogonal.txt slot_random.txt
    self_rank.txt summarize_recent.txt propose_skill_diff.txt
    genes.yaml interest_seed.example.md
  skills/
    taste.md                # human-readable mirror of the active skill version
  matins/
    config.py
    providers/
      base.py anthropic.py openai.py openai_compatible.py search_web.py
      messaging/ base.py telegram.py whatsapp_baileys.py whatsapp_cloud.py
    store/    db.py models.py
    memory/   kernels.py consolidate.py
    generate/ pipeline.py slots.py schema.py novelty.py
    feedback/ capture.py diverge.py
    digest/   render.py
    cli.py                  # run | collect | feedback | consolidate | init-telegram
  data/                     # gitignored: matins.db, logs
  bridges/
    whatsapp_baileys/       # optional Node bridge (only if WhatsApp enabled)
  tests/
```

Core v1 commands: `matins run`, `matins collect`, `matins consolidate` (+ `init-telegram`).
`matins feedback` remains the offline fallback.

---

## 13. Config (model-agnostic)

```yaml
provider:
  name: anthropic            # anthropic | openai | openai_compatible
  model: <model-id>
  base_url: null             # set for local / OpenAI-compatible endpoints
  api_key_env: MATINS_API_KEY

generation:
  n_slots: 4
  temperature: 0.4
  output_language: en        # en | zh | bilingual

novelty:
  search_provider: web       # web | arxiv | none
  k: 5

messaging:
  channel: telegram          # telegram | none | whatsapp_baileys | whatsapp_cloud
  telegram:
    bot_token_env: MATINS_TELEGRAM_TOKEN
    chat_id: "<your_chat_id>"
  collect_delay_hours: 3

memory_kernels:
  - {name: fast, window_days: 7,  stride: 1, aggregator: llm_summarize_recent, feeds: generation}
  - {name: slow, window_days: 75, stride: 1, aggregator: llm_propose_skill_diff, feeds: consolidation}

consolidation:
  cadence_days: 7
  hypothesis_occurrence_threshold: 3
  require_human_approval: true

retrieval:
  sources: []                # arXiv categories, venues, feeds — seeded from your interests
  dedup_against_days: 30

interest_seed_file: prompts/interest_seed.md
```

---

## 14. Scheduling

No daemon in v1. Idempotent entrypoints triggered externally:

```
# Linux/macOS cron
0  8 * * *   cd /path/to/matins && matins run        >> data/cron.log 2>&1
0 11 * * *   cd /path/to/matins && matins collect     >> data/cron.log 2>&1
0  9 * * 1   cd /path/to/matins && matins consolidate  >> data/cron.log 2>&1
```

Windows: Task Scheduler with the same three commands. `matins run` is safe to re-run (one
batch per date). If you reply late, `matins collect` just picks the replies up on its next run.

---

## 15. Risks & mitigations

- **Cold start / data starvation.** Few ideas/day → slow learning. Comments are dense signal;
  manage expectations (months). Unavoidable, not fatal.
- **Goodhart / sycophancy.** Optimizing only "match the user's ranking" degrades ideas into
  flattery. Novelty and quality are first-class objectives, slots C/D are immune to
  fit-pressure by construction, and self-vs-user tau is a *diagnostic*, not a loss to minimize.
- **Skill rot / drift.** Versioning + human-approved consolidation + persistence threshold;
  instant rollback.
- **Reflexive taste drift (Lucas).** Acknowledged, not solved; explore slots + full logging
  make it studyable later.
- **Messaging fragility / ToS (WhatsApp path).** Unofficial WhatsApp can disconnect or get the
  number banned; use a dedicated number, or stay on Telegram. Telegram's Bot API is sanctioned.
- **Provider variance.** Provider + model logged per batch so quality is comparable across them.

---

## 16. Roadmap (post-v1)

- **v1.x:** guided Telegram feedback with inline buttons; medium memory kernel; gene pool grown
  from the log; tiny local web dashboard for browsing the log.
- **v2 — evolutionary engine (your EMD, in embryo):** promote the four slots to a *population*
  of generation strategies ("genomes" = prompt + conditioning config); the human ranking is the
  fitness signal; mutate/crossover the genomes. At that point the daily tool *is* a personal EMD
  / adaptive-governance sandbox, and running it produces logged data for that research line.
- **Structure-internalization:** flip on once §10's backtest says it helps.

---

## 17. Open questions (for calibration)

1. **Output language** — `en`, `zh`, or `bilingual`? (Repo prose is English for
   reproducibility; idea output can differ.)
2. **Reply format** — single-message format in §9.4 (lowest friction) vs. guided
   one-idea-at-a-time buttons (v1.x, nicer but needs a listener)?
3. **Default reference provider** — ship with Anthropic as default + adapters, or document an
   OpenAI-compatible model as the tested reference path for non-Claude users?
4. **Retrieval sources** — which arXiv categories / venues / feeds seed `retrieval.sources`?
   (Needs your actual interest list to be useful on day one.)
5. **Memory kernel defaults** — fast 7d / slow 75d (stride 1, keep the whole window) are guesses; adjust?
6. **Name** — resolved: renamed to **Matins**.
