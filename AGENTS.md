# AGENTS.md

## Purpose

This repository is an Obsidian-based engineering knowledge vault, not a conventional software project.

Keep the following content types distinguishable:

- External sources.
- Personal understanding.
- Project context.
- Experiments and validation.
- Operational runbooks.
- Architecture decisions.
- AI-generated drafts.

## Language and writing

- Follow the user's language unless they request otherwise.
- When editing an existing note, follow its primary language and established terminology.
- When the user language and note language differ, choose the language that best preserves local consistency; ask only when the choice materially affects the deliverable.
- Keep commands, code symbols, configuration keys, API names, paper titles, and technical identifiers in their original language.
- Avoid unnecessary bilingual repetition. Add the original term in parentheses only when it improves precision or searchability.
- Prefer precise, concise, and technically rigorous prose. Avoid filler, exaggerated claims, and unsupported conclusions.
- Preserve existing terminology unless normalization is explicitly requested.

## Repository map

All paths are relative to the Vault root.

```text
00_Home/          Daily dashboards, active work, and quick access.
01_Inbox/         Temporary, unsorted, clipped, or unreviewed material.
10_Projects/      Work tied to a concrete goal, repository, or deliverable.
20_Knowledge/     Reusable and project-independent technical knowledge.
30_Sources/       Papers, documentation, repositories, talks, and standards.
40_Runbooks/      Repeatable operational and troubleshooting procedures.
50_Experiments/   Experiments, benchmarks, and validation records.
60_Decisions/     Architecture Decision Records and important choices.
70_Timeline/      Daily notes, reviews, meetings, and incident timelines.
80_MOCs/          Long-lived topic maps and curated knowledge navigation.
90_Templates/     Obsidian note templates.
91_AI/            Reusable AI prompts, workflows, and evaluations only.
98_Archive/       Deprecated, imported, or inactive material.
99_Assets/        Images, diagrams, attachments, and exported charts.
```

Do not create new top-level directories unless explicitly requested. If placement is ambiguous, use `01_Inbox/Unsorted/` rather than inventing a taxonomy.

Use folders for stable ownership and lifecycle, not every possible topic dimension. Express cross-cutting relationships with frontmatter, Wikilinks, and MOCs.

## Working rules

Before editing:

1. Read this file and any more specific `AGENTS.md` or `AGENTS.override.md` under the target path.
2. Read the target note completely when practical.
3. Inspect nearby notes, MOCs, and sources when they affect the task.
4. Run `git status --short` before broad or multi-file changes.
5. Preserve unrelated user changes.

By default, make the smallest coherent change. Reading, searching, Markdown editing, link repair, minimal frontmatter for new notes, and non-destructive Git inspection are allowed.

Unless explicitly requested, do not:

- Delete, rename, or move notes or attachments.
- Modify more than five files for one writing task.
- Perform repository-wide formatting, metadata migration, or link rewriting.
- Rewrite an entire note when a local edit is sufficient.
- Change `.obsidian/`, plugin settings, `.git/`, `.smart-env/`, caches, workspace files, or generated indexes.
- Add secrets, private endpoints, checkpoints, datasets, full logs, traces, or large generated files.
- Run Git operations that change history, the index, working tree, or remotes, including commit, push, pull, reset, rebase, clean, and force operations.
- Treat AI-generated statements as verified facts or invent citations, results, versions, or source locations.

Destructive or broad operations require explicit authorization and a written scope.

## Content placement

- Choose locations by lifecycle, then ownership, then content type:
  1. Unsorted, incomplete, or unverified → `01_Inbox/`.
  2. Active project-specific → `10_Projects/<Project>/`.
  3. Mature and reusable → the appropriate directory from `20_Knowledge/` through `60_Decisions/`.
  4. Inactive or deprecated → `98_Archive/`, only when authorized.

Keep project-specific experiments, decisions, meetings, and assets inside the project. Extract or link mature, project-independent conclusions into the appropriate reusable area when requested or clearly in scope. Do not move the only copy merely to satisfy the taxonomy.

`00_Home/` answers “What am I working on now?”; `80_MOCs/` answers “What does this topic contain?” Keep task dashboards out of MOCs.

Use `91_AI/` only for reusable prompts, workflows, and evaluations. Classify AI-assisted notes by subject and mark them with metadata.

### Project structure

Each project should have one entry note. Create other subdirectories only when needed:

```text
10_Projects/<Project>/
├── _Project - <Project>.md
├── Notes/
├── Meetings/
├── Experiments/
├── Decisions/
└── Assets/
```

The entry note links to project scope, status, repositories, experiments, decisions, meetings, and derived reusable knowledge.

### Directory growth

Avoid empty or speculative directory trees. Create a category when it has roughly 5–10 items or needs independent browsing; otherwise prefer links, metadata, or a MOC.

## Frontmatter

Use minimal frontmatter for new formal knowledge notes:

```yaml
---
type: knowledge
status: seed
created: YYYY-MM-DD
updated: YYYY-MM-DD
domains: []
aliases: []
source: []
ai_generated: false
reviewed: false
---
```

Supported values:

```text
type: project, knowledge, source, paper, runbook, experiment, decision,
      incident, meeting, daily, moc
status: draft, seed, growing, evergreen, active, paused, completed,
        deprecated, archived
```

Rules:

- Preserve existing fields unless a change is necessary; do not add speculative empty fields.
- Use ISO dates: `YYYY-MM-DD`.
- After substantive AI-generated prose, conclusions, formulas, procedures, or technical changes, set `ai_generated: true` and `reviewed: false` when those fields apply.
- Do not change review fields for typo fixes or mechanical formatting.
- Never set `reviewed: true` on behalf of the user or create artificial confidence scores.

## File naming

Directories use English names and stable numeric prefixes. Notes may use Chinese titles. Prefer descriptive filenames that remain meaningful outside their parent directory.

```text
Project entry:  _Project - <Project Name>.md
Experiment:     EXP-YYYYMMDD-<name>.md
Incident:       INC-YYYYMMDD-<name>.md
Decision:       ADR-NNNN-<name>.md
Paper:          YYYY - <Paper Title>.md
Meeting:        YYYY-MM-DD <Meeting Name>.md
Daily note:     YYYY-MM-DD.md
Weekly note:    YYYY-Www.md
```

Avoid generic names such as `Notes.md`, `Summary.md`, and `Untitled.md`.

For Windows compatibility, do not use `< > : " / \ | ? *`, trailing spaces or periods, or reserved names such as `CON`, `PRN`, `AUX`, and `NUL`. Keep paths reasonably short.

## Obsidian and Markdown

- Preserve YAML frontmatter, heading hierarchy, code fences, formulas, tables, callouts, footnotes, Wikilinks, embeds, and established terminology.
- Prefer one H1 per standalone note, H2 for major sections, short coherent paragraphs, and lists only for enumerable material.
- Prefer Wikilinks: `[[Note Title]]`; use `[[Note Title|display text]]` when needed.
- Preserve embeds such as `![[image.png]]` and do not replace valid Wikilinks with raw paths.
- Do not create links to nonexistent notes unless marked as planned.
- If a rename or move is authorized, update and verify affected links.
- Use MOCs for curated navigation; do not add links solely because notes share a keyword.
- Do not reformat unrelated sections, turn every paragraph into bullets, silently change values or formulas, or remove assumptions and failure conditions.

## Mathematical notation and formulas

Use one consistent explanatory pattern throughout a note:

1. State the quantity being derived and its scope.
2. Define every non-obvious symbol before or immediately after first use.
3. Give tensor shapes, units, dtype byte width, and parallel dimensions when relevant.
4. State assumptions and distinguish per-token, per-parameter, per-layer, per-rank, and global quantities.
5. Present the general formula before substituting numbers.
6. Show key intermediate steps when they help validation.
7. State the result with its unit and applicability boundary.

Formatting rules:

- Use `$...$` for inline math and `$$...$$` for display math.
- Use standard LaTeX notation consistently; do not alternate symbol names for the same quantity.
- Use upright text for labels and units where appropriate, such as `\mathrm{GiB}`.
- Keep prose in the note's primary language while preserving conventional symbols and technical identifiers.
- Do not mix decimal and binary units without stating the conversion, for example GB versus GiB.
- Never change an equation, constant, numerical result, or assumption silently. If correcting one, explain the correction.

Recommended structure:

```markdown
设 $B$ 为批大小，$S$ 为序列长度，$H$ 为隐藏维度，$b$ 为每个元素的字节数。

单层激活内存为：

$$
M = BSHb.
$$

这里的 $M$ 是单卡、单层、未计重计算与临时缓冲区的理论值，单位为字节。
```

## AI content and verification

AI is an assistant, not a source of record.

For substantive AI-generated content:

- Mark it unreviewed and preserve the original question when useful.
- Distinguish verified facts, user-provided facts, source claims, derivations, recommendations, AI inference, and open questions.
- Mark unverifiable claims with `[待核验]`.
- Do not claim that code, commands, formulas, configurations, or results were validated unless they were actually checked.
- Never fabricate citations, quotations, sections, benchmarks, or experimental results.

For an AI draft, prefer:

```markdown
> [!warning]
> 本文包含 AI 生成或改写的实质内容，尚未完成人工核验。
```

Use local source notes and user-provided documents as primary evidence. For external material, record exact titles and attribute claims. Preserve URLs only when provided or verified. Prefer primary sources, and record versions, commits, dates, or access dates when version sensitivity matters. If a source is unavailable, state the limitation.

Link source notes and derived knowledge notes when appropriate.

## Note-type requirements

Include only the sections relevant to the task.

### Technical notes

Consider: motivation, definitions, principles, examples, implementation mapping, cost, assumptions, limitations, related concepts, and evidence.

### Experiments

Record: question or hypothesis, environment, code version, model/configuration, parallelism, data/input, exact command, metrics, observations, interpretation, limitations, artifacts, and follow-up.

Do not generalize beyond recorded conditions. Failed experiments should record why they failed. Keep only small configs, summaries, selected excerpts, and charts in Git.

### Runbooks

Record: applicable scenario, symptoms, fast checks, detailed diagnostics, likely causes, fixes, validation, rollback, and related notes.

Commands must be copyable. Mark destructive steps clearly and avoid claiming universal applicability across versions or environments.

### Decisions

Record: context, selected option, alternatives, trade-offs, rationale, consequences, and review conditions.

## Assets and Obsidian configuration

Choose asset placement by ownership:

- One note or experiment: use a sibling `<note-name>.assets/` directory.
- One project: use `10_Projects/<Project>/Assets/`.
- Reused across multiple projects or topics: use `99_Assets/`.

Move tightly coupled assets with their note when a move is authorized. Do not rename assets casually, duplicate large files, embed base64 data, or modify binaries unless explicitly requested.

Expected plugins:

```text
Core: Properties, Templates, Daily Notes, Backlinks, Outgoing Links,
      Search, Quick Switcher, File Recovery
Community: Copilot for Obsidian, Obsidian Git
Later when needed: Linter, Smart Connections
```

Do not install, remove, enable, disable, or reconfigure plugins unless explicitly requested. Do not modify `.obsidian/workspace.json`, `.obsidian/workspace-mobile.json`, `.obsidian/cache/`, or `.smart-env/`. Keep secrets outside Git.

## Git and validation

This Vault is shared between Windows and Linux.

- Preserve LF line endings and do not rewrite unrelated files.
- Do not modify `.gitattributes` or `.gitignore` without an explicit task.
- Preserve unrelated uncommitted changes.
- Before completing a multi-file task, inspect `git status --short`, `git diff --stat`, `git diff --check`, and the relevant diff.
- If the target already contains unrelated edits, preserve and report them.

## Broad changes

A change is broad if it affects more than five files or includes restructuring, bulk renames, taxonomy changes, frontmatter migration, repository-wide link rewriting, large AI expansion, or plugin configuration.

Before a broad change:

1. Inspect repository structure and Git status.
2. Write a migration plan and exact file scope.
3. Separate structural movement from content rewriting.
4. Work in reviewable batches.
5. Keep the old structure until the new one is validated.
6. Stop when destructive ambiguity remains.

When archiving is authorized, preserve the original semantic location by mirroring it under `98_Archive/`, for example `98_Archive/10_Projects/` or `98_Archive/30_Sources/`. Archiving changes lifecycle, not content type.

## Completion

Before reporting completion, verify that:

- The requested content is in the correct location and unrelated content is preserved.
- Markdown, frontmatter, links, and asset paths remain valid.
- Substantive AI content is marked unreviewed.
- No secrets or large generated files were added.
- The diff contains only intended changes.
- Uncertainty and missing validation are reported.

Report files created, modified, moved, or deleted; major content decisions; unverified claims; and validation performed.

Prefer the safest useful edit over the most ambitious edit.
