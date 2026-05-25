# CLAUDE.md — data/quantapix/

**Kind:** render-cache
**Producer-of-record:** manual mirror (sessions re-render the per-repo
READMEs from `studying/focus-areas.md`, `explaining/outline.md`, etc.;
copy-out to the live `quantapix` GitHub org is external to qagents)
**Consumers:** none in-repo (external: github.com/quantapix)
**Schema:** one `<repo>-public/README.md` per public repo, plus the
parent `README.md` for the org-profile (`quantapix/quantapix`); see § 3.
**Refresh cadence:** weekly during the donation drive
(2026-06-01 → 2026-12-01); slower otherwise
**Write-lock posture:** `.data-write-lock` (manual lane) — re-render
sessions write canonical directly

Project-specific rules for the staging tree that mirrors the public
GitHub `quantapix` organisation. Assumes Claude Code's default guidance
and the repo-root `qagents/CLAUDE.md`. Don't re-litigate those. See
`data/specs/data-charter-2026-05-17.md` § 8.1 — the kind label here is
**render-cache** by default; `external-staging` is reserved if/when the
charter ever flips this entry.

## 1. Role — staging only

`data/quantapix/` is the **staging mirror** of the public org. The user
already maintains the live org and the public repos on GitHub
(`quantapix/quantapix` — the org-profile repo, rendered at the org
page via the user-profile pattern — plus `quantapix/qstudying-public`,
`quantapix/qexplaining-public`, and the four 2026-06-01 launch repos
`qagents-public` / `qnarre-public` / `qresev-public` /
`qdonating-public`); copy-out from here into those repos happens in a
separate terminal and is **not a qagents concern**.

Inside qagents, this directory exists so the public-facing artifacts
sit next to the private working trees they window into — `studying/`,
`explaining/`, `proving/`, `accounting/`, `donating/`, and the
constellation as a whole — and stay consistent as those evolve.

**Naming convention (locked 2026-05-15; product-rename 2026-05-17).**
All public repos under the `quantapix` org carry the `-public` suffix,
with the org-profile repo `quantapix/quantapix` as the only exception.
The suffix marks the repo as the redacted public slice of a private
working tree of the same root name **except** for the two product-named
slices `qnarre-public` (legal-domain proving slice; backs the Qnarre
product) and `qresev-public` (financial-domain accounting slice; backs
the Qresev product). Those two name the product the slice supports
rather than the private subproject directory, because the product name
is what donors and reviewers will see on the live endpoints
(`qnarre.quantapix.com`, `qresev.quantapix.com`). Recorded renames:
`qstudying`→`qstudying-public` and `qexplaining`→`qexplaining-public`
(2026-05-15, suffix unification); `qproving-public`→`qnarre-public`
(2026-05-17, product-rename to match the qresev-public pattern). All
renames happen on the live GitHub side first; this directory mirrors
the renamed shape immediately.

## 2. What this directory is NOT

- **Not a git surface.** Do not `git init` here, do not add this tree
  as a submodule, do not configure remotes on the subdirs themselves.
  The qagents repo tracks these files.
- **Deploy bridge: `git push-quantapix`.** The session-invocable
  wrapper at `~/clone/setup/git-push-quantapix` (symlinked to
  `~/.local/bin/`) is the only allowed push surface — see § 7. It
  keeps its own scratch git clones at `~/.cache/quantapix-push/<repo>/`
  outside the qagents tree and rsyncs from here into them. Replaced
  the prior manual-copy-out workflow on 2026-05-18 when the user
  consolidated `~/clone/{quantapix,qstudying,qexplaining}/` into
  qagents.
- **Not the source of truth for content already pinned elsewhere.**
  The 10 Lean4 focus areas live in `studying/focus-areas.md`; the
  5×10 video plan lives in `explaining/outline.md`. The READMEs here
  are *derived* — public-safe, prose-edited renderings of those
  sources. When the source moves, the README follows; not the other
  way around.

## 3. Layout

```
data/quantapix/
  CLAUDE.md                 (this file; not pushed)
  README.md                 (parent org profile — copies out to quantapix/quantapix)
  LICENSE                   (MIT; org-profile)
  qstudying-public/
    README.md               (qstudying-public; refreshed from studying/focus-areas.md)
    LICENSE                 (MIT; content-class)
  qexplaining-public/
    README.md               (qexplaining-public; refreshed from explaining/outline.md)
    LICENSE                 (MIT; content-class)
  qagents-public/
    README.md               (umbrella; eight-theme rule-set + memory + recall narrative)
    LICENSE                 (Apache-2.0; code lane)
    LICENSE-MIT             (MIT; prose + AI-agentic workflow lane)
    claude-md/              (redacted CLAUDE.md graph — root + per-subproject + data hubs)
      README.md
      root.md / <sub>.md / data/{data,specs,tmp,quantapix}.md  (per Phase 1+ rollout)
    memory/                 (redacted auto-memory topic-file mirror)
      README.md
      MEMORY.md + feedback_*.md + project_*.md + reference_*.md  (per Phase 2+ rollout)
    memsearch/              (curated daily memos; opt-in via `<!-- publish: yes -->`)
      README.md
      <sub>/YYYY-MM-DD.md   (per quarterly sweep)
  qnarre-public/
    README.md               (proving/ slice backing Qnarre: federal civil-rights axiom set)
    LICENSE.txt             (Apache-2.0; code-class)
  qresev-public/
    README.md               (accounting/ slice backing Qresev: financial-strategy axiom set)
    LICENSE.txt             (Apache-2.0; code-class)
  qdonating-public/
    README.md               (donating/ slice: drive + ledgers + weekly digests)
    LICENSE                 (MIT; content-class)
```

The org-profile slot is `data/quantapix/` top-level — README + LICENSE
only. The `-public` subdirs are themselves separate live repos and the
deploy bridge handles them independently (§ 7).

Lives under `data/` rather than at the repo root because it is content
shared across `studying/` and `explaining/` (the source-of-truth working
subprojects), not a working subproject in its own right. Same shape and
policies as `data/specs/`, `data/renders/`, `data/gics/`.

Add a new subdir only when the user has created the corresponding
public repo. New subdirs follow the same shape: a `README.md` as the
canonical entry plus a `LICENSE` for content-class repos (MIT) or
code-class repos (Apache-2.0). `qagents-public/` is the lone exception
— it carries **both** `LICENSE` (Apache, code lane) and `LICENSE-MIT`
(MIT, prose + AI-agentic workflow lane) per the re-charter spec
`data/specs/qagents-public-repo-2026-05-25.md`.

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

The child READMEs are the **public-facing window** into private
qagents activity, refreshed weekly:

- `qstudying-public/README.md` ⮸ `studying/focus-areas.md` — when the
  user re-ranks focus areas, drops a topic, or adds a new active
  thread, re-render the README and the user copies out.
- `qexplaining-public/README.md` ⮸ `explaining/outline.md` — when
  subjects are renamed, P-tags shift, or per-script anchors land,
  re-render.
- `qagents-public/README.md` + `qagents-public/{claude-md,memory,memsearch}/README.md`
  ⮸ root `CLAUDE.md` + every published subproject `CLAUDE.md` — refresh
  when a subproject is added, retired, or a cross-subproject convention
  shifts. Three mirror subtrees feed the umbrella:
  - `claude-md/` ⮸ root + published `<sub>/CLAUDE.md` + four shared-hub
    `data/*/CLAUDE.md` — **weekly** redacted sync via
    `data/quantapix/scripts/sync-mirror.sh`.
  - `memory/` ⮸ `~/.claude/projects/-Users-qpix-clone-qagents/memory/**`
    — **weekly** redacted sync (same tool; topic-file rewrites authored
    per-entry).
  - `memsearch/` ⮸ per-sub `<sub>/.memsearch/memory/**` — **quarterly**
    batch sweep; only memos carrying `<!-- publish: yes -->` flow in.
- `qnarre-public/README.md` ⮸ `proving/CLAUDE.md` +
  `proving/Proving/<Framework>/` — refresh when an axiom set lands or
  a predicate stub shape changes.
- `qresev-public/README.md` ⮸ `accounting/CLAUDE.md` +
  `accounting/Accounting/<Framework>/` — same shape as
  `qnarre-public`.
- `qdonating-public/README.md` ⮸ `donating/drive.md` +
  `donating/ledger/` + `donating/weekly/` — refresh on every monthly
  ledger and every Friday weekly digest during the drive window
  (2026-06-01 → 2026-12-01).

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
grep -RoE "https?://[^ )]+" data/quantapix/ | sort -u

# (c) drift — diff the README against its source (focus-areas.md / outline.md)
# manually; the README is allowed to drop content but not to reorder it
# without a corresponding source edit.
```

(a) must come back empty. (b) is advisory — flag dead URLs but don't
auto-edit. (c) is judgment.

## 7. Deploy bridge — `git push-quantapix`

`~/clone/setup/git-push-quantapix` (symlinked to `~/.local/bin/`) is
the only allowed push surface. Invoke as `git push-quantapix` from
any cwd.

- Scratch git clones live at `~/.cache/quantapix-push/<repo>/`; auto-
  cloned on first run, then `git fetch + reset --hard origin/<default>`
  each subsequent run.
- `rsync -a --exclude=.git` (no `--delete`) carries the source into
  the scratch tree. Live-only files (anything that exists in the
  GitHub repo but not under `data/quantapix/`) survive. Rename/delete
  on the qagents side still requires a manual `git rm` in the scratch
  clone.
- The org-profile slot uses `--exclude='*/'` so only top-level files
  of `data/quantapix/` are synced (the `-public` subdirs are
  themselves separate repos handled by other iterations).
- If the scratch tree has no diff after rsync, the repo is skipped.
  Otherwise: commit `sync from data/quantapix/ <UTC-ts>` and push
  `origin <default>`.
- `--dry-run` rsyncs and shows `git diff --stat` per repo without
  committing/pushing.
- Positional args filter to a subset:
  `git push-quantapix qnarre-public qresev-public`.

GitHub-only. Private working trees (`qagents`, `dot.claude`, etc.)
still use `git push-all` for the github+aws+qblk fan-out — see
`serving/CLAUDE.md` § 9.

## 8. Scope boundary

Same as the rest of the constellation: this subproject does not import
from `analyzing/`, `trading/`, `proving/`, etc., and they do not
import from here. The only inbound dependency is *content* — the
human-readable summaries pulled from `studying/` and `explaining/`
sources.
