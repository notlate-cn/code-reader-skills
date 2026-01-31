# 🧠 Code Deep Understanding Analyzer

<div align="center">

**基于认知科学的源代码深度理解工具 | Cognitive Science-Based Code Analysis Tool**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.1.0-green.svg)](https://github.com/notlate-cn/code-reader-skills/releases)
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
- 🌐 **双语支持** - 完整的中文和英文版本
- ⚡ **三种模式** - Quick/Standard/Deep 满足不同需求

### 🆚 版本对比

<table>
<thead>
  <tr>
    <th>特性</th>
    <th>v1.0 基础版</th>
    <th>v2.1 改进版 ⭐</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>核心目标</strong></td>
    <td>提高阅读效率</td>
    <td>确保真正理解</td>
  </tr>
  <tr>
    <td><strong>分析方法</strong></td>
    <td>逐行解析</td>
    <td>精细询问（强制 WHY）</td>
  </tr>
  <tr>
    <td><strong>分析模式</strong></td>
    <td>单一模式</td>
    <td>Quick/Standard/Deep 三种模式</td>
  </tr>
  <tr>
    <td><strong>验证机制</strong></td>
    <td>❌ 无</td>
    <td>✅ 自我解释 + 应用迁移</td>
  </tr>
  <tr>
    <td><strong>概念处理</strong></td>
    <td>独立解释</td>
    <td>构建概念网络</td>
  </tr>
  <tr>
    <td><strong>理论支撑</strong></td>
    <td>❌ 无</td>
    <td>✅ 认知科学研究</td>
  </tr>
</tbody>
</table>

### 📦 文件结构

```
code-reader-skills/
├── v2.1/                            # 最新版本 ⭐ 推荐
│   ├── code-reader-v2-cn.skill      # 中文 Skill 包
│   ├── code-reader-v2-en.skill      # 英文 Skill 包
│   ├── code-reader-v2-cn-skill.md   # 中文文档
│   └── code-reader-v2-en-skill.md   # 英文文档
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

# 或直接下载 Release
# https://github.com/notlate-cn/code-reader-skills/releases
```

#### 2. 三种分析模式

v2.1 支持三种分析深度，根据场景自动选择：

| 模式 | 耗时 | 适用场景 | 触发词示例 |
|------|------|---------|-----------|
| **Quick** | 5-10 分钟 | 快速浏览、代码审查 | "快速分析"、"简单看看" |
| **Standard** | 15-20 分钟 | 学习理解、技术调研 ⭐ | "深入分析"、"帮我理解" |
| **Deep** | 30+ 分钟 | 面试准备、完全掌握 | "彻底分析"、"我要掌握" |

**默认使用 Standard Mode**

#### 3. 导入到 Claude

1. 访问 [Claude.ai](https://claude.ai)
2. 点击 **Skills** → **Upload Skill**
3. 选择对应的 `.skill` 文件
4. 等待导入完成

#### 4. 开始使用

**Quick Mode 触发示例：**
```
你：快速分析这段代码
[上传或粘贴代码]
```

**Standard Mode 触发示例：**
```
你：深入分析这段代码，我想理解它的设计原理
[上传或粘贴代码]
```

**Deep Mode 触发示例：**
```
你：我需要彻底掌握这个算法，准备面试用
[上传或粘贴代码]
```

### 📊 v2.1 输出结构

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
7. 💻 关键代码深度解析
8. 📦 依赖与使用示例

#### Deep Mode（完全掌握）
[包含 Standard Mode 所有内容，加上]
- 🔗 概念网络图
- 📊 完整执行示例
- 🚀 应用迁移场景（至少 2 个）
- ✅ 质量验证清单

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
- 不写会导致任何人都能访问系统资源

**WHY 选择 JWT 方案？**
- 无状态认证，服务器不需要存储 Session
- 自包含，Token 携带所有必要信息

**WHY 不用 Session？**
- Session 需要服务器存储，不利于水平扩展
- 分布式系统需要共享 Session，增加复杂度

## 算法与理论：JWT (JSON Web Token)

**WHY 使用 JWT？**
- 无状态认证，服务器不需要存储 Session
- 自包含，Token 携带所有必要信息
- 跨域友好，适合微服务架构

**WHY JWT 是安全的？**
- 使用签名验证完整性
- 无法伪造（除非私钥泄露）
```

</details>

#### 示例 2：理解算法复杂度（Deep Mode 应用迁移）

<details>
<summary>点击查看完整示例</summary>

**输入：**
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

**Deep Mode 输出（包含应用迁移）：**

```markdown
## 应用迁移场景：从快排到日志排序

### 场景：对数百万条日志按时间戳排序

**不变的原理：**
- 分治思想：递归分解问题
- Pivot 选择：影响性能的关键
- 原地排序：节省空间

**需要调整的部分：**
```python
# 迁移：外部排序 + timsort 优化
def quicksort_logs(log_file, output_file, memory_limit):
    # WHY 外部排序：数据量超过内存
    chunks = split_file_into_chunks(log_file, memory_limit)

    for chunk in chunks:
        logs = load_chunk(chunk)
        # WHY 用 timsort：日志通常部分有序
        logs.sort(key=lambda log: log.timestamp)
        save_sorted_chunk(chunk, logs)

    merge_sorted_chunks(chunks, output_file)
    return output_file
```

**学到的通用模式：**
- 算法选择取决于数据特征（规模、有序性）
- 基本原理可迁移（分治、比较），但实现需调整
- 超大数据需要外部算法（分块+归并）
```

</details>

### 🔬 研究基础

v2.1 基于以下认知科学研究：

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

根据你的目标选择合适模式。
</details>

<details>
<summary><strong>Q: 可以同时使用中文和英文版本吗？</strong></summary>

**A:** 可以！同时导入两个语言版本，通过对话语言选择使用哪个。
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

#### v2.1.0 (2026-01-31) - 三模式版本

**新增功能：**
- ✨ Quick/Standard/Deep 三种分析模式
- ✨ 智能模式选择机制
- ✨ 优化输出结构

**改进：**
- 📖 更新 README 说明
- 🎯 精简触发词

#### v2.0.0 (2026-01-31) - 改进版发布

**新增功能：**
- ✨ 基于认知科学的分析方法
- ✨ 强制 WHY 询问机制
- ✨ 自我解释测试
- ✨ 概念网络构建
- ✨ 应用迁移测试
- ✨ 质量验证清单

#### v1.0.0 (2026-01-31) - 基础版发布

**核心功能：**
- ✅ 中英文双语支持
- ✅ 8 大分析部分
- ✅ 逐行代码解析

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
- ⚡ **Three Modes** - Quick/Standard/Deep for different needs

### 🆚 Version Comparison

<table>
<thead>
  <tr>
    <th>Feature</th>
    <th>v1.0 Basic</th>
    <th>v2.1 Improved ⭐</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>Core Goal</strong></td>
    <td>Improve reading efficiency</td>
    <td>Ensure true understanding</td>
  </tr>
  <tr>
    <td><strong>Analysis Method</strong></td>
    <td>Line-by-line parsing</td>
    <td>Elaborative interrogation (force WHY)</td>
  </tr>
  <tr>
    <td><strong>Analysis Modes</strong></td>
    <td>Single mode</td>
    <td>Quick/Standard/Deep modes</td>
  </tr>
  <tr>
    <td><strong>Verification</strong></td>
    <td>❌ None</td>
    <td>✅ Self-explanation + Transfer test</td>
  </tr>
  <tr>
    <td><strong>Concept Handling</strong></td>
    <td>Independent explanation</td>
    <td>Build concept network</td>
  </tr>
  <tr>
    <td><strong>Theory Support</strong></td>
    <td>❌ None</td>
    <td>✅ Cognitive science research</td>
  </tr>
</tbody>
</table>

### 🚀 Quick Start

#### 1. Download Skill Files

```bash
# Clone repository
git clone https://github.com/notlate-cn/code-reader-skills.git
cd code-reader-skills
```

#### 2. Three Analysis Modes

| Mode | Duration | Use Case | Trigger Examples |
|------|----------|----------|------------------|
| **Quick** | 5-10 min | Quick browse, code review | "quickly analyze", "briefly look" |
| **Standard** | 15-20 min | Learning, research ⭐ | "deeply analyze", "help me understand" |
| **Deep** | 30+ min | Interview prep, mastery | "thoroughly analyze", "I need to master this" |

**Default: Standard Mode**

#### 3. Import to Claude

1. Visit [Claude.ai](https://claude.ai)
2. Click **Skills** → **Upload Skill**
3. Select corresponding `.skill` file
4. Wait for import completion

### 🔬 Research Foundation

v2.1 is based on the following cognitive science research:

- **[Dunlosky et al. (2013)](https://www.aft.org/ae/fall2013/dunlosky)** - Elaborative interrogation significantly outperforms passive reading
- **[Chi et al. (1994)](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1803_3)** - Self-explainers achieve correct mental models with higher probability
- **[Karpicke & Roediger (2008)](https://science.sciencemag.org/content/319/5865/966)** - Retrieval practice outperforms re-reading by 250%

### 🤝 Contributing

Contributions are welcome! Here's how to participate:

1. **Report Issues** - Provide feedback via [Issues](https://github.com/notlate-cn/code-reader-skills/issues)
2. **Suggest Improvements** - Share your experience and ideas
3. **Submit PRs** - Improve documentation or add new features
4. **Share Cases** - Showcase your use cases

### 📄 License

This project is open-sourced under the [MIT License](LICENSE).

### 🌟 Star History

If this project helps you, please give it a Star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=notlate-cn/code-reader-skills&type=Date)](https://star-history.com/#notlate-cn/code-reader-skills&Date)

---

<div align="center">

**Made with ❤️ and 🧠 for deeper code understanding**

**基于 ❤️ 和 🧠 创建，助力深度理解代码**

[⬆ Back to Top](#-code-deep-understanding-analyzer)

</div>
