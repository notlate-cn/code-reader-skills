# 🧠 Code Deep Understanding Analyzer

<div align="center">

**基于认知科学的源代码深度理解工具 | Cognitive Science-Based Code Analysis Tool**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.3.0-green.svg)](https://github.com/notlate-cn/code-reader-skills/releases)
[![Claude Skills](https://img.shields.io/badge/Claude-Skills-orange.svg)](https://claude.ai)
[![Language](https://img.shields.io/badge/language-中文%20%7C%20English-red.svg)](#)

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 📖 项目简介

这是一套专业的 Claude Skills，帮助开发者**真正理解**源代码，而不只是"看懂"。基于认知科学研究，融合精细询问法、自我解释测试和应用迁移验证，确保深度学习而非产生流畅幻觉。

**核心理念：** 理解为什么 (WHY) > 知道是什么 (WHAT)

### ✨ 核心特性

- 🎯 **精细询问法** - 每个概念强制回答 3 个 WHY
- 🧪 **自我解释测试** - 验证真实理解程度
- 🔗 **概念网络构建** - 建立知识连接，而非孤立记忆
- 🚀 **应用迁移测试** - 检验能否在不同场景应用
- 📚 **学术研究支撑** - 基于 Dunlosky, Chi, Karpicke 等认知科学研究
- ⚡ **四种模式** - Quick/Standard/Deep/Parallel Deep 满足不同需求
- 🤖 **并行分析** - 子 Agent 并行处理，确保大型项目分析深度

### 📦 文件结构

```
code-reader-skills/
├── code-reader-zh/                  # 中文版 🇨🇳
│   ├── SKILL.md                     # Skill 源文件
│   └── code-reader-zh.skill         # Skill 包
│
├── code-reader-en/                  # 英文版 🇺🇸
│   ├── SKILL.md                     # Skill source file
│   └── code-reader-en.skill         # Skill package
│
├── README.md                         # 项目说明（本文件）
└── LICENSE                           # 开源许可证
```

### 🚀 快速开始

#### 1. 下载 Skill 文件

```bash
# 克隆仓库
git clone https://github.com/notlate-cn/code-reader-skills.git
cd code-reader-skills
```

#### 2. 四种分析模式

支持四种分析深度，根据场景自动选择：

| 模式 | 耗时 | 适用场景 | 触发词示例 |
|------|------|---------|-----------|
| **Quick** | 5-10 分钟 | 快速浏览、代码审查 | "快速分析"、"简单看看" |
| **Standard** | 15-20 分钟 | 学习理解、技术调研 ⭐ | "深入分析"、"帮我理解" |
| **Deep** | 30+ 分钟 | 面试准备、完全掌握 | "彻底分析"、"我要掌握" |
| **Parallel Deep** 🚀 | 按项目规模 | 大型项目、复杂代码库 | "并行分析"、"完整项目理解" |

**默认使用 Standard Mode**

**Parallel Deep Mode 专为大型项目（>5000 行）设计，使用子 Agent 并行处理各章节，确保每个章节都有足够深度。**

#### 3. 安装 Skill

**方式一：复制到 Claude 目录（推荐）**

```bash
# 中文版
mkdir -p ~/.claude/skills
cp -r code-reader-zh ~/.claude/skills

# 英文版
mkdir -p ~/.claude/skills
cp -r code-reader-en ~/.claude/skills
```

**方式二：直接在对话中使用**

```bash
# 在分析代码前，直接粘贴 skill 内容
cat code-reader-zh/SKILL.md | pbcopy  # macOS
# 或者在对话中输入 @文件路径
```

**方式三：使用 API 的开发者**

如果你通过 Claude API 调用，可以将 skill 内容作为 System Prompt：

```python
import anthropic

# 读取 skill 内容（中文版）
with open('code-reader-zh/SKILL.md', 'r', encoding='utf-8') as f:
    skill_content = f.read()

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-20250514",  # 或使用最新可用模型
    system=skill_content,  # 将 skill 作为系统提示
    messages=[
        {"role": "user", "content": "深入分析这段代码：\n\n[你的代码]"}
    ]
)
```

#### 4. 开始使用

**Quick Mode 触发示例：**
```
你：/code-reader-v2-cn 快速分析这段代码
[上传或粘贴代码]
```

**Standard Mode 触发示例：**
```
你：/code-reader-v2-cn 深入分析这段代码，我想理解它的设计原理
[上传或粘贴代码]
```

**Deep Mode 触发示例：**
```
你：/code-reader-v2-cn 我需要彻底掌握这个算法，准备面试用
[上传或粘贴代码]
```

**Parallel Deep Mode 触发示例：**
```
你：/code-reader-v2-cn 并行分析这个大型项目
[上传或粘贴代码，或提供项目路径]
```

### 📊 输出结构

#### Quick Mode（快速概览）
1. 👀 快速概览
2. 📝 功能说明
3. 📐 核心算法/设计
4. 💻 关键代码段
5. 📦 依赖关系
6. 🎯 快速使用示例

#### Standard Mode（标准理解）⭐
1. 🧪 理解验证状态
2. 👀 快速概览
3. 🎯 背景与动机（精细询问）
4. 📌 核心概念说明
5. 📐 算法与理论
6. 🏗️ 设计模式
7. 💻 关键代码深度解析（场景化注释）
8. 📦 依赖与使用示例

#### Deep Mode（完全掌握）
[包含 Standard Mode 所有内容，加上]
- 🔗 概念网络图
- 📊 完整执行示例（多场景追踪）
- 🚀 应用迁移场景（至少 2 个）
- ✅ 质量验证清单
- 📝 渐进式生成（确保深度）
- 💾 直接写入文件（Token 优化）

#### Parallel Deep Mode（大型项目专用）🚀
[包含 Deep Mode 所有内容，采用并行架构]
- 🤖 主协调 Agent：框架生成、任务分发、结果汇总
- ⚡ 并行子 Agents：8 个章节同时处理
- 📊 独立上下文：每个章节都有充分深度
- 🔄 自动汇总：生成最终完整文档
- 📁 文件结构：`00-框架.json` + `tasks/` + `chapters/` → `最终文档.md`

### 💡 使用示例

#### 示例 1：学习 JWT 认证（Standard Mode）

<details>
<summary>点击查看完整示例</summary>

**输入：**
```python
def authenticate_user(username, password):
    user = db.find_user(username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return generate_token(user.id)
    return None
```

**输出（节选）：**
```markdown
## 背景与动机分析

**WHY 需要这段代码？**
- 系统需要验证用户身份，保护未授权访问

**WHY 选择 JWT 方案？**
- 无状态认证，服务器不需要存储 Session

**WHY 不用 Session？**
- Session 需要服务器存储，不利于水平扩展
```

</details>

### 🔬 研究基础

基于以下认知科学研究：

- **[Dunlosky et al. (2013)](https://www.aft.org/ae/fall2013/dunlosky)** - 精细询问法显著优于被动阅读
- **[Chi et al. (1994)](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1803_3)** - 自我解释者获得正确心智模型的概率更高
- **[Karpicke & Roediger (2008)](https://science.sciencemag.org/content/319/5865/966)** - 检索练习优于重复阅读 250%

### 🤝 贡献指南

欢迎贡献！以下是参与方式：

1. **报告问题** - 通过 [Issues](https://github.com/notlate-cn/code-reader-skills/issues) 反馈
2. **提出建议** - 分享你的使用体验和改进想法
3. **提交 PR** - 改进文档或添加新功能
4. **分享案例** - 展示你的使用案例

### ❓ 常见问题

<details>
<summary><strong>Q: 三种模式有什么区别？</strong></summary>

**A:**
- **Quick**：快速了解代码结构和功能，5-10 分钟
- **Standard**：理解 WHY 和设计思路，15-20 分钟（推荐）
- **Deep**：应用迁移测试 + 质量验证，30+ 分钟
</details>

<details>
<summary><strong>Q: 支持哪些编程语言？</strong></summary>

**A:** 支持所有主流语言：
- Python, JavaScript, TypeScript, Java, C++, Go, Rust
- 以及各种框架和库
</details>

<details>
<summary><strong>Q: 如何判断我真正理解了？</strong></summary>

**A:** 使用 Deep Mode 的"四能"测试：
1. ✅ 能否不看代码解释设计思路？
2. ✅ 能否独立实现类似功能？
3. ✅ 能否应用到不同场景？
4. ✅ 能否向他人清晰解释？
</details>

### 📝 更新日志

#### v2.3.0 (2026-02-07)

**新增功能：**
- 🚀 Parallel Deep Mode（并行深度模式）- 专为大型项目设计
- 🤖 子 Agent 并行架构 - 确保每个章节都有足够深度
- 📊 主协调 Agent - 框架生成、任务分发、结果汇总
- 🔄 完整实现指南 - Task tool 调用示例

**改进：**
- 📖 更新为四种分析模式
- 🎯 大型项目（>5000 行）自动建议并行模式
- 📝 新增 CLAUDE.md 仓库指南

**解决问题：**
- ✅ 解决大型项目分析时章节内容过浅的问题
- ✅ 并行处理提高效率，同时确保深度

#### v2.2.0 (2026-02-04)

**新增功能：**
- ✨ 场景化注释风格（场景 N / 步骤 N）
- ✨ Token 优化策略（直接写入文件）
- ✨ 渐进式生成（Deep Mode 专用）
- ✨ 多语言注释支持规范

**改进：**
- 📦 优化文档结构，删除重复内容
- 📖 统一章节深度自检标准
- 🎯 完善执行流示例格式

**优化：**
- 🔧 文件大小：1500 → 1342 行 (-10.5%)
- ⚡ 删除子 Agent 使用规范
- 📝 合并重复的模式选择说明

#### v2.1.0 (2026-01-31)

**新增功能：**
- ✨ Quick/Standard/Deep 三种分析模式
- ✨ 智能模式选择机制
- ✨ 优化输出结构

**改进：**
- 📦 独立中英文目录结构
- 📖 更新 README 说明
- 🎯 精简触发词

### 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

### 🌟 Star History

如果这个项目对你有帮助，请给个 Star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=notlate-cn/code-reader-skills&type=Date)](https://star-history.com/#notlate-cn/code-reader-skills&Date)

### 📧 联系方式

- 问题反馈：[GitHub Issues](https://github.com/notlate-cn/code-reader-skills/issues)
- 讨论交流：[GitHub Discussions](https://github.com/notlate-cn/code-reader-skills/discussions)

---

## English

### 📖 Project Introduction

A professional Claude Skills set that helps developers **truly understand** source code, not just "get it." Based on cognitive science research, integrating elaborative interrogation, self-explanation testing, and application transfer verification to ensure deep learning rather than fluency illusion.

**Core Philosophy:** Understanding WHY > Knowing WHAT

### ✨ Key Features

- 🎯 **Elaborative Interrogation** - Force answering 3 WHYs for each concept
- 🧪 **Self-Explanation Test** - Verify true understanding
- 🔗 **Concept Network Construction** - Build knowledge connections, not isolated memories
- 🚀 **Application Transfer Test** - Examine if applicable in different scenarios
- 📚 **Academic Research Support** - Based on Dunlosky, Chi, Karpicke's cognitive science research
- 🌐 **Bilingual Support** - Complete Chinese and English versions
- ⚡ **Four Modes** - Quick/Standard/Deep/Parallel Deep for different needs
- 🤖 **Parallel Analysis** - Sub-agents process chapters in parallel, ensuring depth for large projects

### 📦 File Structure

```
code-reader-skills/
├── code-reader-zh/                  # Chinese version 🇨🇳
│   ├── SKILL.md                     # Skill source file
│   └── code-reader-zh.skill         # Skill package
│
├── code-reader-en/                  # English version 🇺🇸
│   ├── SKILL.md                     # Skill source file
│   └── code-reader-en.skill         # Skill package
│
├── README.md                         # Project documentation (this file)
└── LICENSE                           # Open source license
```

### 🚀 Quick Start

#### 1. Download Skill Files

```bash
# Clone repository
git clone https://github.com/notlate-cn/code-reader-skills.git
cd code-reader-skills
```

#### 2. Four Analysis Modes

| Mode | Duration | Use Case | Trigger Examples |
|------|----------|----------|------------------|
| **Quick** | 5-10 min | Quick browse, code review | "quickly analyze", "briefly look" |
| **Standard** | 15-20 min | Learning, research ⭐ | "deeply analyze", "help me understand" |
| **Deep** | 30+ min | Interview prep, mastery | "thoroughly analyze", "I need to master this" |
| **Parallel Deep** 🚀 | Scales with project | Large projects, complex codebases | "parallel analyze", "complete project understanding" |

**Default: Standard Mode**

**Parallel Deep Mode is designed for large projects (>5000 lines), using sub-agents to process chapters in parallel, ensuring sufficient depth for each chapter.**

#### 3. Install Skill

**Method 1: Copy to Claude directory (Recommended)**

```bash
# English version
mkdir -p ~/.claude/skills
cp -r code-reader-en ~/.claude/skills

# Chinese version
mkdir -p ~/.claude/skills
cp -r code-reader-zh ~/.claude/skills
```

**Method 2: Use directly in conversation**

```bash
# Paste skill content before analyzing code
cat code-reader-en/SKILL.md | pbcopy  # macOS
# Or use @file-path in conversation
```

**Method 3: For API Users**

If you're using Claude API, pass skill content as System Prompt:

```python
import anthropic

# Read skill content (English version)
with open('code-reader-en/SKILL.md', 'r', encoding='utf-8') as f:
    skill_content = f.read()

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=skill_content,  # Use skill as system prompt
    messages=[
        {"role": "user", "content": "Deeply analyze this code:\n\n[Your code]"}
    ]
)
```

#### 4. Getting Started

**Quick Mode Trigger Example:**
```
You: /code-reader-v2-en Quickly analyze this code
[Upload or paste code]
```

**Standard Mode Trigger Example:**
```
You: /code-reader-v2-en Deeply analyze this code, I want to understand its design principles
[Upload or paste code]
```

**Deep Mode Trigger Example:**
```
You: /code-reader-v2-en I need to thoroughly master this algorithm for interview preparation
[Upload or paste code]
```

**Parallel Deep Mode Trigger Example:**
```
You: /code-reader-v2-en Parallel analyze this large project
[Upload or paste code, or provide project path]
```

### 📊 Output Structure

#### Quick Mode
1. Quick Overview
2. Functionality Description
3. Core Algorithm/Design
4. Key Code Segments
5. Dependencies
6. Quick Usage Example

#### Standard Mode ⭐
1. Understanding Verification Status
2. Quick Overview
3. Background & Motivation (Elaborative Interrogation)
4. Core Concepts
5. Algorithm & Theory
6. Design Patterns
7. Key Code Deep Analysis (Scenario-based comments)
8. Dependencies & Usage Examples

#### Deep Mode
[All Standard Mode content, plus]
- Concept Network Diagram
- Complete Execution Examples (Multi-scenario tracking)
- Application Transfer Scenarios (at least 2)
- Quality Verification Checklist
- Progressive Generation (Ensure depth)
- Direct File Writing (Token optimized)

#### Parallel Deep Mode 🚀 (For Large Projects)
[All Deep Mode content, with parallel architecture]
- Master Coordinator Agent: Framework generation, task dispatch, result aggregation
- Parallel Sub-Agents: 8 chapters processed simultaneously
- Independent Context: Each chapter has sufficient depth
- Auto-Aggregation: Generate final complete document
- File Structure: `00-framework.json` + `tasks/` + `chapters/` → `final-document.md`

### 🔬 Research Foundation

Based on the following cognitive science research:

- **[Dunlosky et al. (2013)](https://www.aft.org/ae/fall2013/dunlosky)** - Elaborative interrogation significantly outperforms passive reading
- **[Chi et al. (1994)](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1803_3)** - Self-explainers achieve correct mental models with higher probability
- **[Karpicke & Roediger (2008)](https://science.sciencemag.org/content/319/5865/966)** - Retrieval practice outperforms re-reading by 250%

### 🤝 Contributing

Contributions are welcome! Here's how to participate:

1. **Report Issues** - Provide feedback via [Issues](https://github.com/notlate-cn/code-reader-skills/issues)
2. **Suggest Improvements** - Share your experience and ideas
3. **Submit PRs** - Improve documentation or add new features
4. **Share Cases** - Showcase your use cases

### ❓ FAQ

<details>
<summary><strong>Q: What's the difference between four modes?</strong></summary>

**A:**
- **Quick**: Fast understanding of structure and functionality, 5-10 min
- **Standard**: Understand WHY and design rationale, 15-20 min (Recommended)
- **Deep**: Application transfer testing + quality verification, 30+ min
- **Parallel Deep**: For large projects (>5000 lines), parallel chapter processing with guaranteed depth
</details>

<details>
<summary><strong>Q: Which programming languages are supported?</strong></summary>

**A:** All mainstream languages:
- Python, JavaScript, TypeScript, Java, C++, Go, Rust
- And various frameworks and libraries
</details>

<details>
<summary><strong>Q: How do I know if I truly understand?</strong></summary>

**A:** Use Deep Mode's "four abilities" test:
1. ✅ Can you explain the design rationale without looking at code?
2. ✅ Can you independently implement similar functionality?
3. ✅ Can you apply it to different scenarios?
4. ✅ Can you clearly explain it to others?
</details>

### 📝 Changelog

#### v2.3.0 (2026-02-07)

**New Features:**
- 🚀 Parallel Deep Mode - Designed for large projects
- 🤖 Sub-Agent parallel architecture - Ensures sufficient depth for each chapter
- 📊 Master coordinator agent - Framework generation, task dispatch, result aggregation
- 🔄 Complete implementation guide - Task tool usage examples

**Improvements:**
- 📖 Updated to four analysis modes
- 🎯 Auto-suggest parallel mode for large projects (>5000 lines)
- 📝 Added CLAUDE.md repository guide

**Problems Solved:**
- ✅ Fixed shallow chapter content issue in large project analysis
- ✅ Parallel processing improves efficiency while ensuring depth

#### v2.2.0 (2026-02-04)

**New Features:**
- ✨ Scenario-based comment style (Scenario N / Step N)
- ✨ Token optimization strategy (direct file writing)
- ✨ Progressive generation (Deep Mode only)
- ✨ Multi-language comment support standards

**Improvements:**
- 📦 Optimized document structure, removed duplicates
- 📖 Unified chapter depth self-check standards
- 🎯 Enhanced execution flow example format

**Optimizations:**
- 🔧 File size: 1500 → 1342 lines (-10.5%)
- ⚡ Removed Sub-Agent usage guidelines
- 📝 Merged duplicate mode selection sections

#### v2.1.0 (2026-01-31)

**New Features:**
- ✨ Quick/Standard/Deep analysis modes
- ✨ Smart mode selection mechanism
- ✨ Optimized output structure

**Improvements:**
- 📦 Separate Chinese/English directory structure
- 📖 Updated README documentation
- 🎯 Refined trigger words

### 📄 License

This project is open-sourced under the [MIT License](LICENSE).

### 🌟 Star History

If this project helps you, please give it a Star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=notlate-cn/code-reader-skills&type=Date)](https://star-history.com/#notlate-cn/code-reader-skills&Date)

### 📧 Contact

- Bug Reports: [GitHub Issues](https://github.com/notlate-cn/code-reader-skills/issues)
- Discussions: [GitHub Discussions](https://github.com/notlate-cn/code-reader-skills/discussions)

---

<div align="center">

**Made with ❤️ and 🧠 for deeper code understanding**

**基于 ❤️ 和 🧠 创建，助力深度理解代码**

[⬆ Back to Top](#-code-deep-understanding-analyzer)

</div>
