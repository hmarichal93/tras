---
name: technical-writing
description: Write documentation that is easy to read and act on. ALWAYS use when creating or editing any README, Markdown file (.md), design/decision doc, experiment report, project status page, how-to, CLAUDE.md, issue or PR description. Covers structure (answer first), plain language, scannable Markdown, runnable examples, and a pre-publish checklist.
---

# Technical writing: docs people can read in one pass

A document succeeds when the intended reader can **find what they need in under a minute and act
on it without asking**. Write for that reader, not for completeness.

## 1. Before writing: answer three questions

1. **Who reads it?** (new contributor, the future you, a reviewer, a user of the tool)
2. **What do they need to do after reading?** (run it, decide, review, reproduce, continue a task)
3. **What is the one-sentence answer?** If you can't write it, you're not ready to write the doc.

Pick the document type (§5) from the answer to question 2.

## 2. Structure: answer first, details later

- **Lead with the conclusion** (BLUF: bottom line up front). The first 3 lines say what this is,
  the key result or decision, and what to do next.
- **Progressive disclosure:** summary → how to use → how it works → details/reference. Readers
  stop when they have enough; make every stopping point useful.
- **One idea per section, one topic per paragraph.** Paragraphs ≤ 4 sentences.
- **Headings are labels a reader scans for:** task-based ("Run the evaluation") or
  question-based ("Why not DETR?"), never vague ("Notes", "Misc", "Discussion").
- **Put reference material in tables or appendices**, not in the narrative.
- **Link, don't duplicate.** Each fact lives in one place (single source of truth); other docs link to it.

## 3. Sentences: plain, concrete, active

- Short sentences (aim ≤ 25 words). Active voice: "The script writes X", not "X is written".
- Plain words: "use" not "utilize", "because" not "due to the fact that".
- **Concrete over abstract:** numbers with units and context ("F = 0.953 vs 0.951 baseline on
  UruDendro1 test"), not "slightly better".
- **Define each term once**, at first use, then use exactly that term. Don't alternate synonyms
  ("disk" / "section" / "slice") for the same thing; if they are the same, say so once.
- Expand acronyms at first use. Avoid unexplained jargon; when a concept matters, cover what it is,
  why it is used, and the alternative (one line each is enough).
- State uncertainty explicitly: measured / analytic estimate / from literature (unverified).
- No filler ("It is worth noting that", "As mentioned above", "In order to").

## 4. Markdown style

- **One `#` H1** (the title). Sentence-case headings. Max depth `###` in most docs.
- **Bullets for parallel items, numbered lists for sequences**, tables for comparisons across ≥ 2
  attributes. Keep list items grammatically parallel.
- **Bold sparingly**: only for the 1–3 things a skimmer must not miss per section.
- **Code blocks are copy-paste runnable**, fenced with a language tag (` ```bash `, ` ```python `),
  from the repo root, with no `$` prompts and no placeholder you didn't explain (`<dir>` is
  defined right next to the block).
- Inline code for paths, commands, identifiers, config keys: `experiments/r1_matching_check.py`.
- **Relative links** inside the repo (`[spec](docs/problem-formulation.md)`); descriptive link
  text, never "click here".
- **Images:** alt text + a caption sentence saying what to look at. Keep files small (< 300 KB).
- Tables: ≤ ~7 columns; units in headers (`gap p5 px`); align numbers to the same precision.
- Callouts for the critical warning only (`> **Warning:** …`). One per section at most.
- Dates absolute (2026-10-10), never "last week". Version/commit when results depend on it.

## 5. Templates by document type

**README (entry point to a repo or folder)**
1. Title + one-sentence purpose.
2. Status / key result (1–3 bullets) and links to the main docs.
3. Quickstart: install + one command that produces something, with expected output.
4. Usage (common tasks), project layout (short tree or table), data requirements.
5. How to contribute / conventions, license/citation.
Keep it under ~1 screen before the first scroll for sections 1–3.

**Decision document (ADR / GO–NO-GO)**
Decision box (the call, scope, what it gates) → evidence summary table (claim → test → result →
script) → context and options considered → analysis per acceptance criterion → risks/failure
modes → reversal criteria → consequences for other tasks → references.

**Experiment report**
Question → answer (one line, with the number) → setup (data, split, config, command, commit) →
results table (+ per-sample spread) → what surprised us → limitations → next step.

**How-to / runbook**
Goal → prerequisites → numbered steps (one action each, with the command) → how to verify it
worked → troubleshooting (symptom → cause → fix).

**Status / handoff page**
Last updated date → task board table (task, status, outcome, next action) → decision log
(date, decision, evidence, link) → current task plan → open questions.

**Issue / PR description**
What and why in 2–3 lines → what changed (bullets) → evidence/how tested → what reviewers should
check → `Closes #N`.

## 6. Pre-publish checklist

1. Can a reader get the main point from the first 3 lines?
2. Does every section heading tell the skimmer what's inside?
3. Any paragraph > 4 sentences or sentence > ~30 words? Split it.
4. Every number has units, a comparison point, and a source (script/paper)?
5. Every command runs as written from a clean clone? (Run it if possible.)
6. Every term defined once and used consistently? Acronyms expanded?
7. Any duplicated content that should be a link instead?
8. Links and image paths resolve? (`grep -o '](.*)'` and check.)
9. Is anything stale (dates, statuses, "TODO")? Remove or date it.
10. Would cutting 20% lose anything the reader needs? If not, cut it.

When editing an existing doc, keep its structure and voice unless they block readability; fix
the sections you touch and leave a consistent result.
