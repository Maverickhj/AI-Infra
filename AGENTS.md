# AI-Infra Knowledge Base

## Mission

This repository is an Obsidian knowledge base for an LLM systems engineer, with Transformer engineering as its center of gravity. It is a persistent, compounding artifact: new sources and useful answers must be integrated into existing maps, concepts, runbooks, and decisions instead of remaining isolated chat output.

The vault has three layers:

1. **Raw sources** — immutable inputs under `30_Sources/Raw/`.
2. **Wiki** — maintained notes under `00_Home/` through `80_MOCs/`.
3. **Schema** — this file, templates, and the maintenance workflow under `91_AI/`.

## Repository map

```text
00_Home/          Home page and active navigation.
01_Inbox/         Unsorted questions, clips, and temporary notes.
10_Projects/      Context tied to a repository, model, or deliverable.
20_Knowledge/     Reusable technical concepts and mental models.
30_Sources/       Raw inputs and source notes; Raw is immutable.
40_Runbooks/      Repeatable diagnosis and operating procedures.
50_Experiments/   Reproducible experiments and validation records.
60_Decisions/     Architecture Decision Records.
70_Timeline/      Append-only activity log and incident timelines.
80_MOCs/          Curated Maps of Content.
90_Templates/     Note templates.
91_AI/            AI maintenance workflows and reusable prompts.
98_Archive/       Deprecated material; move only with permission.
99_Assets/        Shared images and exported artifacts.
index.md          Complete content-oriented catalog.
```

Do not introduce a new top-level directory unless the user requests it. Use folders for stable ownership or lifecycle; use frontmatter, Wikilinks, and MOCs for cross-cutting dimensions.

## Default workflow

### Ingest

1. Preserve the original input in `30_Sources/Raw/` or link to its stable location. Never rewrite a raw source.
2. Create or update one source note describing claims, applicability, version/date, and limitations.
3. Integrate reusable conclusions into existing knowledge notes. Prefer updating a hub over creating a near-duplicate page.
4. Add deliberate cross-links to prerequisites, implementation paths, experiments, and runbooks.
5. Update `index.md` and append one entry to `70_Timeline/log.md`.

### Query

1. Read `index.md`, then the relevant MOC and notes.
2. Answer with the shortest sufficient producer-to-consumer chain, including process/rank/device boundaries when relevant.
3. Separate evidence, derivation, inference, recommendation, and open questions.
4. When an answer is reusable, file it into the wiki and update links and the log.

### Lint

Check for broken links, orphan notes, duplicate concepts, stale or contradictory claims, missing evidence, inconsistent terminology, and runbooks without validation or rollback steps. Never resolve a contradiction by silently deleting one side.

## Engineering writing contract

- Follow the user's language; keep code symbols, configuration keys, paper titles, and API names in their original language.
- Prefer precise Chinese prose and searchable English aliases in frontmatter.
- For code-path questions, trace the concrete producer, transformations, transport/process boundary, and final consumer.
- For distributed systems, distinguish global, node, process, rank, thread, logical CUDA device, and physical GPU.
- For tensor reasoning, define shape, dtype, byte width, parallel dimension, and whether a value is per-token, per-layer, per-rank, or global.
- For configuration, distinguish code default, enable condition, precedence, runtime value, and workload recommendation.
- For comparisons, distinguish similar functional layers from numerical or algorithmic equivalence.
- For runtime diagnosis, record the innermost traceback, phase, rank, collective, pod/retry state, and memory evidence before naming a root cause.

## Evidence and review state

Substantive AI-generated content must use:

```yaml
ai_generated: true
reviewed: false
```

Never mark content reviewed on the user's behalf. Use these labels in prose when needed:

- **已验证** — directly supported by a cited source, inspected code, or recorded experiment.
- **推导** — follows from stated assumptions or formulas.
- **推断** — plausible but not directly proven.
- **建议** — an action choice, not a fact.
- **待核验** — evidence is missing or version-sensitive.

Do not invent citations, versions, commands, measurements, or repository paths. Static inspection is not GPU/runtime validation.

## Minimal frontmatter

```yaml
---
type: knowledge
status: seed
created: YYYY-MM-DD
updated: YYYY-MM-DD
domains: []
aliases: []
tags: []
source: []
ai_generated: true
reviewed: false
---
```

Supported `type`: `project`, `knowledge`, `source`, `paper`, `runbook`, `experiment`, `decision`, `incident`, `meeting`, `daily`, `moc`.

Supported `status`: `draft`, `seed`, `growing`, `evergreen`, `active`, `paused`, `completed`, `deprecated`, `archived`.

Keep frontmatter minimal: omit fields that do not add information. Use ISO dates. Preserve existing fields when editing.

## Note patterns

- Knowledge: question → mental model → mechanics → implementation mapping → cost/trade-offs → failure modes → related notes.
- Runbook: symptoms → fast triage → evidence collection → decision tree → fixes → validation → rollback.
- Experiment: hypothesis → environment/commit → topology/config → command → metrics → observations → interpretation → limitations → artifacts.
- Decision: context → decision → alternatives → consequences → review conditions.
- Source: bibliographic identity → claims → evidence strength → applicability → conflicts → derived notes.

Use one H1 per standalone note. Prefer Wikilinks. Do not create a link to a nonexistent note unless it is explicitly marked `planned`.

## Safety and change discipline

- Preserve unrelated user changes and inspect `git status --short` before broad edits.
- Do not change `.obsidian/`, `.claudian/`, plugin files, secrets, large logs, datasets, checkpoints, or generated caches unless explicitly requested.
- Do not delete, rename, archive, or bulk-rewrite notes without explicit scope.
- Prefer a small coherent update. When a request spans more than five files, state the planned scope before editing and validate the final link graph and diff.

## Completion checklist

- The new knowledge is integrated into a MOC and `index.md`.
- Evidence and uncertainty are visible.
- Cross-links exist for actual conceptual or operational relationships.
- Substantive AI writing remains unreviewed.
- `70_Timeline/log.md` records the operation.
- Markdown links, frontmatter, and the scoped Git diff have been checked.
