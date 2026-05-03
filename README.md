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

## Team

Two contributors. One software developer; one expert AI assistant.

- **Imre Kifor** — sole developer. Engineering, statute encoding,
  predicate authoring, kernel design, product surfaces.
- **Claude Code (Opus)** — third teammate. Per-task subagents under
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

## Public repos under this organisation

| Repo | Role | Cadence |
|---|---|---|
| [`qstudying`](https://github.com/quantapix/qstudying) | Lean4 expert-track focus areas + OSS contribution targets that back `proving/` (Qnarre) and `accounting/` (Qresev). The "where to read, what to write down" companion. | weekly-refreshed |
| [`qexplaining`](https://github.com/quantapix/qexplaining) | 50-video explainer arc (5 topics × 10 subjects), narrated by **Janet**, brand-synced with the two product sites. The "convey how it works" companion, scoped to the pre-beta window. | weekly-refreshed |

The two products' own repositories will join this list when they ship.

## Contact

[`quantapix@gmail.com`](mailto:quantapix@gmail.com) is the only contact channel.

## License

Each public repo carries its own LICENSE (MIT for the three
scripts-only repos: `quantapix`, `qstudying`, `qexplaining`).
