---
name: code-reader-cn
description: 专业的源代码阅读分析助手（中文版），帮助快速理解代码的背景、架构、核心原理和实现细节。当用户需要阅读、理解、分析源代码文件或代码片段时使用，特别是当用户说"帮我读这个代码"、"分析这段代码"、"解释这个源文件"、"这个函数/类/模块是做什么的"时触发。
---

# 源代码阅读分析器 (Code Reader - 中文版)

专业的源代码分析工具，将复杂代码转化为清晰、结构化的技术文档。

## 核心目标

**提高阅读效率，保留核心细节。** 用通俗易懂的方式解释代码，同时保持技术准确性。

## 分析流程

按顺序执行以下步骤，除非明确不适用：

### 1. 代码概览 (Code Overview)

首先快速扫描代码，识别：
- 编程语言 (Programming Language) 和版本
- 文件类型（单文件 vs 多文件项目）
- 代码规模（行数、复杂度）
- 主要依赖库 (Dependencies)

### 2. 背景与用途 (Context & Purpose)

回答"为什么存在这段代码？"

**必须包含：**
- **业务背景** — 解决什么实际问题？
- **技术背景** — 在什么技术栈/框架下工作？
- **核心作用** — 主要功能是什么？
- **适用场景** — 什么时候使用这段代码？

**示例格式：**
```markdown
## 背景与用途

**业务背景：** 这是一个用户认证 (User Authentication) 模块，用于处理用户登录和权限验证。

**技术背景：** 基于 Express.js 框架，使用 JWT (JSON Web Token) 进行无状态认证。

**核心作用：** 提供用户注册、登录、token 验证和刷新功能。

**适用场景：** Web 应用需要安全的用户身份认证时使用。
```

### 3. 架构设计 (Architecture Design)

说明代码的整体结构和组织方式。

**必须包含：**
- **文件/模块结构** — 代码如何组织？
- **核心组件** — 主要的类 (Class)、函数 (Function)、模块 (Module) 是什么？
- **数据流向** — 数据如何在组件间流动？
- **设计模式** — 使用了哪些设计模式？（如使用，需标注并解释）

**设计模式标注格式：**
```markdown
### 设计模式

- **单例模式 (Singleton Pattern)** — `DatabaseConnection` 类确保全局只有一个数据库连接实例
  - 参考：[Singleton Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/singleton)
- **工厂模式 (Factory Pattern)** — `createUser()` 函数根据类型创建不同的用户对象
  - 参考：[Factory Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/factory-method)
```

### 4. 核心原理 (Core Principles)

解释代码背后的关键技术原理和算法。

**必须包含：**
- **核心算法** — 使用了什么算法？时间/空间复杂度？
- **关键技术** — 依赖什么技术原理？
- **实现策略** — 为什么这样实现？有什么权衡？

**算法标注格式：**
```markdown
### 核心算法

- **快速排序 (Quick Sort)** — 用于用户列表排序
  - **时间复杂度 (Time Complexity):** 平均 O(n log n)，最坏 O(n²)
  - **空间复杂度 (Space Complexity):** O(log n)
  - **参考资料:**
    - [Quick Sort - Wikipedia](https://en.wikipedia.org/wiki/Quicksort)
    - [Quick Sort Visualization](https://www.toptal.com/developers/sorting-algorithms/quick-sort)

- **LRU 缓存 (LRU Cache)** — 缓存最近访问的数据
  - **数据结构:** 哈希表 + 双向链表 (HashMap + Doubly Linked List)
  - **时间复杂度:** Get/Put 都是 O(1)
  - **参考资料:**
    - [LRU Cache - LeetCode](https://leetcode.com/problems/lru-cache/)
```

### 5. 详细实现 (Detailed Implementation)

**核心原则：** 用简洁语言 + 详细注释 + 实际示例讲解关键代码段。

#### 5.1 关键代码段识别

选择最重要的代码段（通常 3-5 段），包括：
- 核心算法实现
- 关键业务逻辑
- 复杂的数据处理
- 易混淆的边界条件

#### 5.2 逐行解析格式

对每个关键代码段，使用以下格式：

```markdown
### 代码段：[功能名称]

**作用：** [一句话说明这段代码做什么]

**原始代码：**
\`\`\`python
def authenticate_user(username, password):
    user = db.find_user(username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return generate_token(user.id)
    return None
\`\`\`

**逐行解析：**

\`\`\`python
def authenticate_user(username, password):
    # 定义用户认证函数，接收用户名和密码作为参数
    
    user = db.find_user(username)
    # 从数据库查找用户
    # db.find_user() 返回 User 对象或 None
    
    if not user:
        return None
    # 用户不存在时，返回 None 表示认证失败
    
    if verify_password(password, user.password_hash):
        # 验证密码是否匹配
        # verify_password() 使用 bcrypt 比对明文密码和哈希值
        
        return generate_token(user.id)
        # 密码正确，生成 JWT token 并返回
        # generate_token() 创建包含 user.id 的加密 token
        
    return None
    # 密码错误，返回 None 表示认证失败
\`\`\`

**执行流程示例：**

输入：`username="alice", password="secret123"`

1. `db.find_user("alice")` → 返回 User(id=1, username="alice", password_hash="$2b$...")
2. `user` 存在，继续执行
3. `verify_password("secret123", "$2b$...")` → 返回 True
4. `generate_token(1)` → 返回 "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
5. 函数返回 token 字符串

**关键要点：**
- 密码永远不明文存储，使用 bcrypt 哈希
- 认证失败时一律返回 None，不暴露失败原因（安全考虑）
- token 包含用户 ID，用于后续请求验证
```

### 6. 依赖关系 (Dependencies)

说明代码的外部依赖。

```markdown
## 依赖关系

### 外部库
- **express** (v4.18.0) — Web 框架 (Web Framework)
- **bcrypt** (v5.1.0) — 密码哈希 (Password Hashing)
- **jsonwebtoken** (v9.0.0) — JWT token 生成与验证

### 内部模块
- `database.js` — 数据库连接和查询
- `utils/crypto.js` — 加密工具函数
- `middleware/auth.js` — 认证中间件
```

### 7. 使用示例 (Usage Examples)

提供完整的、可运行的示例。

```markdown
## 使用示例

### 示例 1：用户登录

\`\`\`javascript
// 导入模块
const auth = require('./auth');

// 调用认证函数
const token = await auth.authenticate_user('alice', 'secret123');

if (token) {
    console.log('登录成功！Token:', token);
    // 输出：登录成功！Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
} else {
    console.log('用户名或密码错误');
}
\`\`\`

**执行结果：**
- 如果认证成功，返回 JWT token 字符串
- 如果失败，返回 `null`

### 示例 2：验证 Token

\`\`\`javascript
const userId = auth.verify_token(token);

if (userId) {
    console.log('Token 有效，用户 ID:', userId);
    // 输出：Token 有效，用户 ID: 1
} else {
    console.log('Token 无效或已过期');
}
\`\`\`
```

### 8. 潜在问题与改进建议 (Issues & Improvements)

**可选部分** — 仅在代码存在明显问题时包含。

```markdown
## 潜在问题与改进建议

### 性能问题
- **问题：** 每次请求都查询数据库，可能成为性能瓶颈
- **建议：** 添加 Redis 缓存层，缓存用户信息

### 安全问题
- **问题：** Token 没有设置过期时间
- **建议：** 在 `generate_token()` 中添加 `expiresIn: '1h'` 参数

### 代码质量
- **问题：** 缺少输入验证
- **建议：** 添加用户名和密码格式验证
```

## 输出格式

分析结果必须输出为 **Markdown 文档**，结构清晰，易于阅读。

### 标准输出模板

```markdown
# [代码名称] 源代码分析

## 1. 代码概览
- 编程语言：
- 文件类型：
- 代码规模：
- 主要依赖：

## 2. 背景与用途
- 业务背景：
- 技术背景：
- 核心作用：
- 适用场景：

## 3. 架构设计
- 文件/模块结构：
- 核心组件：
- 数据流向：
- 设计模式：

## 4. 核心原理
- 核心算法：
- 关键技术：
- 实现策略：

## 5. 详细实现
### [代码段 1]
- 作用：
- 原始代码：
- 逐行解析：
- 执行流程示例：
- 关键要点：

### [代码段 2]
...

## 6. 依赖关系
- 外部库：
- 内部模块：

## 7. 使用示例
- 示例 1：
- 示例 2：

## 8. 潜在问题与改进建议（可选）
```

## 质量标准

每份代码分析必须满足：

### 完整性
- ✅ 涵盖所有 8 个核心部分（第 8 部分可选）
- ✅ 关键代码段有详细的逐行解析
- ✅ 算法和设计模式都有明确标注和参考资料

### 准确性
- ✅ 技术术语准确，关键词语标注英文
- ✅ 代码解析正确，没有误导性内容
- ✅ 示例可运行，输出结果真实

### 可读性
- ✅ 语言通俗易懂，避免过度专业术语
- ✅ 结构清晰，使用标题和列表组织内容
- ✅ 重要概念用**粗体**标注，英文术语用括号标注

### 实用性
- ✅ 执行流程示例具体，便于理解
- ✅ 使用示例完整可运行
- ✅ 关键要点突出重点，避免冗余

## 特殊情况处理

### 多文件项目
1. 先分析项目整体结构
2. 识别入口文件和核心模块
3. 按模块分别分析，说明模块间关系
4. 提供模块调用流程图（用 Markdown 表格或文字描述）

### 复杂算法
1. 先用自然语言描述算法思路
2. 提供算法时间/空间复杂度
3. 逐步拆解算法步骤
4. 用具体数据展示执行过程
5. 附上权威参考资料链接

### 不熟悉的技术栈
1. 先查询并说明技术栈背景
2. 解释关键概念和术语
3. 重点分析代码逻辑，而非框架细节
4. 提供该技术栈的学习资源

## 分析前检查清单

开始分析前，确认：
- [ ] 已理解用户的具体需求（理解什么？解决什么问题？）
- [ ] 已识别代码的编程语言和框架
- [ ] 已确定分析范围（单个函数 vs 整个项目）
- [ ] 已准备好相关的参考资料链接
