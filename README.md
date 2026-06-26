# quantapix

> Engineering practice. Two products, one method, two domains.

The method: wrap a Lean4 kernel around LLM-backed predicate functions.
Predicates return Booleans with evidence. The kernel composes the proof.
The two products are **Qnarre** — an axiomatic verifier for legal
complaints — and **Qresev** — an axiomatic evaluator for stocks and
portfolios. Same kernel, different statutes, different OHLCV. Both ship
as early-beta on or about **2026-06-01**, each with a live trace:
every `True` or `False` the system returns is anchored to a quoted
source.

- **<https://quantapix.com>** — engineering output (code, schemas, traces, framework specs).
- **<https://femfas.net>** — motivational record. The system that was
  specced from the inside — federal civil-rights litigation, pro se,
  against an institutional defendant. The long version. quantapix.com
  is the short one.
- **<https://www.youtube.com/@Quantapix>** — the video channel. A 5×10
  explainer arc (five topics, ten subjects each), AI-narrated over
  animated cards and graphics; every video anchors back to a real file in
  the working repository. First episodes landing June 2026.

## Team

Two contributors. One software developer; one expert AI assistant.

- **Imre Kifor** — sole developer. Engineering, statute encoding,
  predicate authoring, kernel design, product surfaces.
- **Claude Code (Opus)** — second teammate. Per-task subagents under
  written `CLAUDE.md` contracts; persistent semantic memory across
  sessions; auditable tool use. The assistant writes, reviews, and
  refactors against the same kernel and predicates the developer reads.

> Real, life-altering problems are excellent system specs. They refuse
> to let you cheat. Don't reduce a claim to opinion. Reduce it to a
> proof. Have a kernel that does no I/O check the proof. Have
> predicates — small, replaceable, audited — read the natural language
> the kernel won't touch. The result is a derivation, not a narrative;
> it survives a hostile reader because there is nothing to disagree
> with that isn't checkable.

## Why these public repos exist

AI is rapidly commoditizing software code. For a sole-developer
practice — where coordination between developers is not a concern —
what is worth sharing publicly is no longer the code itself. It is the
**AI-assisted workflows** that produce the code, and specifically the
workflows that:

1. **support learning** — where to read, what to skip, what to write
   down, what to drop into the agent's prompt;
2. **attract attention** — concise public artifacts that make the
   thesis legible to readers who haven't lived the engineering;
3. **convey how a product works** (or how it should be used) once the
   product is shipping.

Until Qnarre and Qresev start early beta on or about 2026-06-01, the
public surface of this organisation is the first two of the three:
**learning** and **attention**. Both are weekly-refreshed curated
windows into the private working repository's activities.

The 2026-06-01 → 2026-12-01 window is also the window of a **public
donation drive** backing the framework. The drive funds four exclusive-use
buckets (a Max20 subscription whose included $200/month SDK credit pool
covers all programmatic LLM calls — see
[`qdonating-public`](https://github.com/quantapix/qdonating-public)
for the per-tier breakdown — a legal-research MCP subscription, AWS
billing, and federal docketing fees). The drive does not gate the work;
it caps the rate at which the four fixed costs eat into non-engineering
time. The work, including the public-repo refresh cadence, runs whether
or not the drive funds it.

## Help axiomatize the U.S. Code, in the age of AI

The framework's largest open program **aims to** produce a kernel-checked
Lean4 encoding of the operative content of the United States Code — all 54
titles, on the order of tens of thousands of sections — built redundantly
by independent agent teams and reconciled in the kernel. The encoding is
**asserted-and-pending**: a growing fraction of the Code is encoded and
kernel-scored today, with the full corpus an open goal, not a finished
artifact. The program is **open to outside contributors**: it works over
public federal statutes only, so it carries no privacy-floor surface, and
it is the natural place to build the project in the open. The project
today is a single developer working with AI assistance, now opening this
effort to contributors. Pick up a section, a predicate, or a title —
start at [`qnarre-public`](https://github.com/quantapix/qnarre-public)
(CONTRIBUTING + good-first-issues) and the
[`qagents-public`](https://github.com/quantapix/qagents-public)
Discussions.

## Public repos under this organisation

| Repo                                                                    | Role                                                                                                                                                                                                                    | Cadence                          |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| [`qagents-public`](https://github.com/quantapix/qagents-public)         | Umbrella — the redacted `CLAUDE.md` graph that governs AI-assisted authorship across the framework. The "how the practice runs" companion.                                                                              | weekly-refreshed from 2026-06-01 |
| [`qdonating-public`](https://github.com/quantapix/qdonating-public)     | Redacted slice of the public donation drive: pitch, deliverable promises, monthly ledgers, weekly digests. Six-month window 2026-06-01 → 2026-12-01.                                                                    | weekly-refreshed from 2026-06-01 |
| [`qexplaining-public`](https://github.com/quantapix/qexplaining-public) | 50-video explainer arc (5 topics × 10 subjects), AI-narrated, brand-synced with the two product sites. The "convey how it works" companion, scoped to the pre-beta window.                                    | weekly-refreshed                 |
| [`qnarre-public`](https://github.com/quantapix/qnarre-public)           | Redacted legal-domain slice backing the **Qnarre** product: Lean4 axiom set for civil RICO + §§ 1981/1983/1985(3) + Title VI, with predicate stubs, a thin driver, and the full-U.S.-Code axiomatization program.       | weekly-refreshed from 2026-06-01 |
| [`qresev-public`](https://github.com/quantapix/qresev-public)           | Redacted financial-domain umbrella backing the **Qresev** product: the Lean4 evaluator (TREND / MOMENTUM / OPTIONS-RISK / SECTOR / DRAWDOWN) plus the market-inspection and portfolio-management surfaces that feed it. | weekly-refreshed from 2026-06-01 |
| [`qstudying-public`](https://github.com/quantapix/qstudying-public)     | Lean4 expert-track focus areas + OSS contribution targets that back the proving and accounting kernels. The "where to read, what to write down" companion.                                                              | weekly-refreshed                 |

The two products' own repositories will join this list when they ship.

## Contact

[`quantapix@gmail.com`](mailto:quantapix@gmail.com) is the only contact channel.

## License

Each public repo carries its own LICENSE. The split is content-class
vs. code-class:

- **MIT** (content-class — prose, plans, READMEs that happen to embed
  short code snippets): `quantapix` (this org-profile repo),
  `qstudying-public`, `qexplaining-public`, `qdonating-public`.
- **Apache-2.0** (code-class — Lean axiom sets, predicate stubs,
  drivers, framework code redacted for public consumption):
  `qagents-public`, `qnarre-public`, `qresev-public`.

Content-class repos ship a top-level `LICENSE`; code-class repos ship
a top-level `LICENSE.txt`. The filename split is load-bearing inside
the staging tree; the live repos preserve it on push.
