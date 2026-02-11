# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Skills** repository containing bilingual (Chinese/English) code analysis tools based on cognitive science research. The skills help developers achieve deep understanding of source code through elaborative interrogation, self-explanation testing, and retrieval practice.

**Core Philosophy:** Understanding WHY > Knowing WHAT

## Repository Structure

```
deep-code-understanding-skill/
├── code-reader-zh/          # Chinese version skill
│   ├── SKILL.md             # Source skill definition (primary)
│   └── code-reader-zh.skill # Packaged skill file
├── code-reader-en/          # English version skill
│   ├── SKILL.md             # Source skill definition (primary)
│   └── code-reader-en.skill # Packaged skill file
├── README.md                # Bilingual project documentation
├── LICENSE                  # MIT License
└── CLAUDE.md                # This file
```

## Key Architecture

### Skill Definition Format

Skills use YAML frontmatter followed by Markdown content:

```yaml
---
name: code-reader-v2-cn
description: [Trigger words and brief description]
---

# [Content follows...]
```

### Three Analysis Modes

| Mode | Duration | Use Case | Key Features |
|------|----------|----------|--------------|
| **Quick** | 5-10 min | Code review, quick browse | Overview, dependencies, key segments |
| **Standard** | 15-20 min | Learning, research | WHY analysis, design patterns, algorithm deep dive |
| **Deep** | 30+ min | Mastery, interview prep | Application transfer, quality verification, progressive generation |

### Core Analysis Framework

The skills implement a 9-step research-driven workflow:

1. **Quick Overview** - Language, scale, dependencies, code type
2. **Elaborative Interrogation** - 3 WHYs for background/motivation
3. **Concept Network Construction** - Concept relationships with WHYs
4. **Algorithm & Theory Analysis** - Complexity, rationale, references
5. **Design Pattern Recognition** - WHY used, alternatives, implications
6. **Line-by-Line Analysis** - Scenario-based comments with execution tracking
7. **Application Transfer Testing** - At least 2 different scenarios
8. **Dependencies & Usage** - WHY for each dependency
9. **Quality Verification** - Self-assessment checklist

## Common Development Tasks

### Testing Skill Changes

```bash
# Copy skill to local Claude directory for testing
cp -r code-reader-zh ~/.claude/skills/

# Or for English version
cp -r code-reader-en ~/.claude/skills/
```

### Creating Skill Package (.skill file)

```bash
# The .skill file is created by zipping the skill directory
# Format: directory with skill-name matching YAML frontmatter
```

### Publishing New Version

1. Update version in `SKILL.md` frontmatter
2. Update README.md changelog
3. Update both `code-reader-zh/SKILL.md` and `code-reader-en/SKILL.md` for consistency
4. Commit with semantic version message

## Important Development Constraints

### Synchronized Updates

**Both language versions MUST be updated together:**
- `code-reader-zh/SKILL.md` and `code-reader-en/SKILL.md` should have identical structure
- When adding features, update both versions
- Version numbers should match across languages

### Token Optimization Strategy

The skills use a "write directly to file" strategy to save tokens:
- Full analysis written to file via Write tool
- Only brief summary output to conversation
- Deep Mode uses progressive generation for long documents

### Comment Style Convention

Code analysis uses scenario-based/step-based comments:
- `# 场景 N: [description]` / `// Scenario N: [description]` - for conditional branches
- `# 步骤 N: [description]` / `// Step N: [description]` - for sequential flow
- `# 此时: xxx` / `// At this point: xxx` - for variable state tracking

## Research Foundation

The skills are based on cognitive science research:

- **Dunlosky et al. (2013)** - Elaborative interrogation outperforms passive reading
- **Chi et al. (1994)** - Self-explainers build correct mental models more reliably
- **Karpicke & Roediger (2008)** - Retrieval practice beats re-reading by 250%

References included in skill output should link to authoritative sources.

## File Naming Conventions

- Generated analysis documents: `[code-name]-深度分析.md` or `[code-name]-deep-analysis.md`
- Multi-file projects: `[project]-overview.md` + `[module]-analysis.md`
- Always use descriptive, searchable names

## Understanding Verification

The skills include a verification gate at the start of analysis:

| Core Concept | Self-Explanation | Understanding WHY | Application Transfer | Status |
|--------------|------------------|-------------------|---------------------|---------|
| [Concept]    | ✅/❌            | ✅/❌             | ✅/❌                | [Status]|

This table should appear at the beginning of Standard/Deep mode outputs.
