# CLAUDE.md — quantapix/

Project-specific rules for the staging tree that mirrors the public
GitHub `quantapix` organisation. Assumes Claude Code's default guidance
and the repo-root `qagents/CLAUDE.md`. Don't re-litigate those.

## 1. Role — staging only

`quantapix/` is the **staging mirror** of the public org. The user
already maintains the live org and the public repos on GitHub
(`quantapix/.github`, `quantapix/qstudying`, `quantapix/qexplaining`,
and any future products); copy-out from here into those repos happens
in a separate terminal and is **not a qagents concern**.

Inside qagents, this directory exists so the public-facing artifacts
(currently three READMEs) sit next to the private working tree they
window into — `studying/` and `explaining/` — and stay consistent with
the rest of the constellation as it evolves.

## 2. What this directory is NOT

- **Not a git surface.** Do not `git init` here, do not add this tree
  as a submodule, do not configure remotes. The qagents repo already
  tracks these files. The public org has its own repos that the user
  syncs by hand.
- **Not a deploy target.** No CI hook, no rsync, no GitHub-CLI push
  from this session. If a future workflow needs automated sync, the
  user will introduce it deliberately; until then, copy-out is manual.
- **Not the source of truth for content already pinned elsewhere.**
  The 10 Lean4 focus areas live in `studying/focus-areas.md`; the
  5×10 video plan lives in `explaining/outline.md`. The READMEs here
  are *derived* — public-safe, prose-edited renderings of those
  sources. When the source moves, the README follows; not the other
  way around.

## 3. Layout

```
quantapix/
  CLAUDE.md            (this file)
  README.md            (parent org / .github profile README)
  qstudying/
    README.md          (qstudying public repo README)
  qexplaining/
    README.md          (qexplaining public repo README)
```

Add a new subdir only when the user has created the corresponding
public repo. New subdirs follow the same shape: a README.md as the
canonical entry, optional `LICENSE` if the user wants license tracking
in qagents too (today they don't — license files are managed live in
the public repos).

## 4. Voice + redaction guardrails

These files will be copied to **public** GitHub repos. Treat them like
any other public-site copy:

- Engineer-debugging voice (no activism, no exhortations, no
  rhetorical questions, no marketing exclamations). Cross-reference
  `feedback_engineer_not_activist_voice.md` and
  `feedback_no_political_activism_tone.md`.
- Sole contact: `quantapix@gmail.com`. Sole GitHub destination:
  `https://github.com/quantapix` (and per-repo URLs under it). No
  other email, phone, address, or per-developer URL.
  Cross-reference `feedback_public_contact_and_github.md`.
- No qagents-rooted internal paths (`legal/hub/...`, `appealing/...`,
  `pleading/...`, full `proving/Proving/<F>/<File>.lean` paths).
  Subproject *names* like `proving/`, `accounting/`, `verifying/`,
  `evaluating/` are fine as engineering-surface references; rooted
  paths inside them are not. Cross-reference
  `feedback_no_qagents_paths_in_letters.md`.
- No opposing-party names, no children's names, no private addresses,
  no political-identity rhetoric. The specific blocklist is held in
  the parent qagents tree (`documenting/letters/REDACTION.md` and the
  patterns enforced by `documenting/scripts/check_redactions.py`) — not
  enumerated here so this file is itself safe to mirror.
- No "Silcrow." Cross-reference `feedback_quantapix_naming.md`.
- Brand-sync: same calm declarative cadence as quantapix.com and
  femfas.net. Pull paragraph rhythm from
  `designing/web/src/content/copy.ts` (`about` block) when in doubt.

## 5. Refresh cadence

The two child READMEs are the **public-facing window** into private
qagents activity, refreshed weekly:

- `qstudying/README.md` ⮸ `studying/focus-areas.md` — when the user
  re-ranks focus areas, drops a topic, or adds a new active thread,
  re-render the README and the user copies out.
- `qexplaining/README.md` ⮸ `explaining/outline.md` — when subjects
  are renamed, P-tags shift, or per-script anchors land, re-render.

The parent `README.md` only changes when the team, the thesis, the
public-repo roster, or the contact channel changes — slower cadence.

## 6. Verifiable hand-off

Before telling the user "ready to copy out", verify:

```bash
# (a) readability — sweep against the constellation's private redaction
# blocklist (see documenting/letters/REDACTION.md and the patterns in
# documenting/scripts/check_redactions.py). The blocklist itself is
# deliberately not duplicated here so this file remains safe to mirror.

# (b) link integrity — every referenced URL resolves on the public side
grep -RoE "https?://[^ )]+" quantapix/ | sort -u

# (c) drift — diff the README against its source (focus-areas.md / outline.md)
# manually; the README is allowed to drop content but not to reorder it
# without a corresponding source edit.
```

(a) must come back empty. (b) is advisory — flag dead URLs but don't
auto-edit. (c) is judgment.

## 7. Scope boundary

Same as the rest of the constellation: this subproject does not import
from `analyzing/`, `trading/`, `proving/`, etc., and they do not
import from here. The only inbound dependency is *content* — the
human-readable summaries pulled from `studying/` and `explaining/`
sources.
