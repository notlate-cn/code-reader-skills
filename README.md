# 🧠 Code Deep Understanding Analyzer

<div align="center">

**基于认知科学的源代码深度理解工具 | Cognitive Science-Based Code Analysis Tool**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/notlate-cn/code-reader-skills/releases)
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

### 🆚 版本对比

<table>
<thead>
  <tr>
    <th>特性</th>
    <th>v1.0 基础版</th>
    <th>v2.0 改进版 ⭐</th>
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
  <tr>
    <td><strong>适用场景</strong></td>
    <td>快速了解代码</td>
    <td>深度学习和掌握</td>
  </tr>
</tbody>
</table>

### 📦 文件结构

```
code-reader-skills/
├── v1.0-basic/                      # 基础版本
│   ├── code-reader-cn.skill         # 中文 Skill 包
│   ├── code-reader-en.skill         # 英文 Skill 包
│   ├── code-reader-cn-skill.md      # 中文文档
│   └── code-reader-en-skill.md      # 英文文档
│
├── v2.0-improved/                   # 改进版本 ⭐ 推荐
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

#### 2. 选择合适的版本

**推荐流程：**

```
新手入门 → v1.0 基础版
  ↓
逐步熟悉
  ↓
深度学习 → v2.0 改进版 ⭐
```

**选择指南：**

| 场景 | 推荐版本 | 文件 |
|------|---------|------|
| 快速浏览代码库 | v1.0 | `code-reader-cn.skill` |
| 代码审查 | v1.0 | `code-reader-en.skill` |
| 学习编程技术 | v2.0 ⭐ | `code-reader-v2-cn.skill` |
| 技术面试准备 | v2.0 ⭐ | `code-reader-v2-en.skill` |
| 研究开源项目 | v2.0 ⭐ | `code-reader-v2-cn.skill` |

#### 3. 导入到 Claude

1. 访问 [Claude.ai](https://claude.ai)
2. 点击 **Skills** → **Upload Skill**
3. 选择对应的 `.skill` 文件
4. 等待导入完成

#### 4. 开始使用

**v1.0 触发示例：**
```
你：帮我分析这段代码
[上传或粘贴代码]
```

**v2.0 触发示例：**
```
你：深入分析这段代码，我想理解它的设计原理
[上传或粘贴代码]
```

### 💡 使用示例

#### 示例 1：学习 JWT 认证（v2.0）

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

**v2.0 输出（节选）：**

```markdown
## 理论基础：JWT (JSON Web Token)

**WHY 使用 JWT？**
- 无状态认证，服务器不需要存储 Session
- 自包含，Token 携带所有必要信息
- 跨域友好，适合微服务架构

**WHY JWT 是安全的？**
- 使用签名验证完整性
- 无法伪造（除非私钥泄露）

**WHY JWT 有局限性？**
- 无法主动失效（需维护黑名单）
- Token 体积较大（Base64 编码增加约 33%）

## 应用迁移场景：API 密钥验证

**不变的原理：**
- 验证调用方身份的核心流程
- 哈希存储凭证

**需要修改的部分：**
- 单参数（API 密钥）而非用户名+密码
- 增加权限范围 (scope)

## 质量验证清单

- [x] 能否不看代码解释 JWT 原理？
- [x] 能否说出为什么选择 JWT？
- [x] 能否应用到其他场景？
```

</details>

#### 示例 2：理解算法复杂度（v2.0）

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

**v2.0 输出（节选）：**

```markdown
## 算法分析：快速排序

**WHY 选择快速排序？**
- 平均 O(n log n)，实际应用中通常最快
- 原地排序，空间效率高
- 缓存友好，访问局部性好

**WHY 可接受最坏 O(n²)？**
- 最坏情况概率极低
- 可通过随机化 Pivot 避免

**WHY 不用归并/堆排序？**
- 归并：需要 O(n) 额外空间
- 堆排：缓存性能差

**什么时候会退化？**
- 已排序输入 → 用随机化解决
- Pivot 选择不当 → 用三数取中优化
```

</details>

### 📊 v2.0 输出结构

完整分析文档包含 **9 个核心部分**：

1. 🧪 **理解验证状态** - 自我解释测试结果
2. 👀 **快速概览** - 语言、规模、依赖
3. 🎯 **背景与动机** - 3 层 WHY 分析
4. 🔗 **概念网络图** - 概念清单 + 关系矩阵
5. 📐 **算法与理论** - 复杂度 + WHY + 参考
6. 🏗️ **设计模式** - WHY 使用 + WHY 有效
7. 💻 **关键代码** - 逐行 WHY + 执行示例
8. 🚀 **应用迁移** - 至少 2 个不同场景
9. ✅ **质量验证** - 自我评估清单

### 🔬 研究基础

v2.0 基于以下认知科学研究：

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
<summary><strong>Q: 我应该用 v1.0 还是 v2.0？</strong></summary>

**A:** 取决于你的目标：
- **快速了解** → v1.0
- **深度掌握** → v2.0 ⭐
- 也可以都导入，根据需求切换
</details>

<details>
<summary><strong>Q: 可以同时使用中文和英文版本吗？</strong></summary>

**A:** 可以！同时导入两个语言版本，通过对话语言选择使用哪个。
</details>

<details>
<summary><strong>Q: v2.0 的分析会很长吗？</strong></summary>

**A:** 是的，v2.0 更详细，但这是有意为之：
- 更多 WHY 解释 → 深层理解
- 应用迁移示例 → 检验掌握
- 质量验证清单 → 确保完整
</details>

<details>
<summary><strong>Q: 支持哪些编程语言？</strong></summary>

**A:** 支持所有主流语言：
- Python, JavaScript, TypeScript, Java, C++, Go, Rust
- 以及各种框架和库
</details>

<details>
<summary><strong>Q: 如何判断我真正理解了？</strong></summary>

**A:** 使用 v2.0 的"四能"测试：
1. ✅ 能否不看代码解释设计思路？
2. ✅ 能否独立实现类似功能？
3. ✅ 能否应用到不同场景？
4. ✅ 能否向他人清晰解释？
</details>

### 📝 更新日志

#### v2.0.0 (2026-01-31) - 改进版发布

**新增功能：**
- ✨ 基于认知科学的分析方法
- ✨ 强制 WHY 询问机制
- ✨ 自我解释测试
- ✨ 概念网络构建
- ✨ 应用迁移测试
- ✨ 质量验证清单

**改进：**
- 📚 添加学术研究支撑
- 📖 完善文档和示例
- 🎯 优化触发机制

#### v1.0.0 (2026-01-31) - 基础版发布

**核心功能：**
- ✅ 中英文双语支持
- ✅ 8 大分析部分
- ✅ 逐行代码解析
- ✅ 算法和设计模式标注
- ✅ 多文件项目分析

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

### 🆚 Version Comparison

<table>
<thead>
  <tr>
    <th>Feature</th>
    <th>v1.0 Basic</th>
    <th>v2.0 Improved ⭐</th>
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
  <tr>
    <td><strong>Use Cases</strong></td>
    <td>Quick code overview</td>
    <td>Deep learning and mastery</td>
  </tr>
</tbody>
</table>

### 🚀 Quick Start

#### 1. Download Skill Files

```bash
# Clone repository
git clone https://github.com/notlate-cn/code-reader-skills.git
cd code-reader-skills

# Or download Release directly
# https://github.com/notlate-cn/code-reader-skills/releases
```

#### 2. Choose Appropriate Version

**Recommended Flow:**

```
Beginner → v1.0 Basic
  ↓
Gradually Familiar
  ↓
Deep Learning → v2.0 Improved ⭐
```

#### 3. Import to Claude

1. Visit [Claude.ai](https://claude.ai)
2. Click **Skills** → **Upload Skill**
3. Select corresponding `.skill` file
4. Wait for import completion

### 🔬 Research Foundation

v2.0 is based on the following cognitive science research:

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

[⬆ 回到顶部](#-code-deep-understanding-analyzer)

</div>
