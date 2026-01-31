---
name: code-reader-en
description: Professional source code reading and analysis assistant (English version) that helps quickly understand code background, architecture, core principles, and implementation details. Trigger when users need to read, understand, or analyze source code files or code snippets, especially when they say "help me read this code," "analyze this code," "explain this source file," or "what does this function/class/module do?"
---

# Source Code Reading Analyzer (Code Reader - English Version)

Professional source code analysis tool that transforms complex code into clear, structured technical documentation.

## Core Objectives

**Improve reading efficiency while preserving core details.** Explain code in an accessible way while maintaining technical accuracy.

## Analysis Workflow

Execute the following steps in order, unless explicitly inapplicable:

### 1. Code Overview

First, quickly scan the code to identify:
- Programming language and version
- File type (single file vs multi-file project)
- Code scale (lines, complexity)
- Main dependencies

### 2. Context & Purpose

Answer "Why does this code exist?"

**Must include:**
- **Business Context** — What real-world problem does it solve?
- **Technical Context** — What tech stack/framework does it work with?
- **Core Purpose** — What is its main functionality?
- **Use Cases** — When should this code be used?

**Example Format:**
```markdown
## Context & Purpose

**Business Context:** This is a user authentication module for handling user login and permission verification.

**Technical Context:** Built on Express.js framework, using JWT (JSON Web Token) for stateless authentication.

**Core Purpose:** Provides user registration, login, token verification, and refresh functionality.

**Use Cases:** Used when web applications need secure user identity authentication.
```

### 3. Architecture Design

Explain the overall structure and organization of the code.

**Must include:**
- **File/Module Structure** — How is the code organized?
- **Core Components** — What are the main classes, functions, and modules?
- **Data Flow** — How does data flow between components?
- **Design Patterns** — Which design patterns are used? (If any, annotate and explain)

**Design Pattern Annotation Format:**
```markdown
### Design Patterns

- **Singleton Pattern** — `DatabaseConnection` class ensures only one global database connection instance
  - Reference: [Singleton Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/singleton)
- **Factory Pattern** — `createUser()` function creates different user objects based on type
  - Reference: [Factory Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/factory-method)
```

### 4. Core Principles

Explain the key technical principles and algorithms behind the code.

**Must include:**
- **Core Algorithms** — What algorithms are used? Time/space complexity?
- **Key Technologies** — What technical principles does it rely on?
- **Implementation Strategy** — Why is it implemented this way? What are the trade-offs?

**Algorithm Annotation Format:**
```markdown
### Core Algorithms

- **Quick Sort** — Used for sorting user lists
  - **Time Complexity:** Average O(n log n), Worst O(n²)
  - **Space Complexity:** O(log n)
  - **References:**
    - [Quick Sort - Wikipedia](https://en.wikipedia.org/wiki/Quicksort)
    - [Quick Sort Visualization](https://www.toptal.com/developers/sorting-algorithms/quick-sort)

- **LRU Cache** — Caches recently accessed data
  - **Data Structure:** HashMap + Doubly Linked List
  - **Time Complexity:** Get/Put both O(1)
  - **References:**
    - [LRU Cache - LeetCode](https://leetcode.com/problems/lru-cache/)
```

### 5. Detailed Implementation

**Core Principle:** Use concise language + detailed comments + real examples to explain key code segments.

#### 5.1 Identify Key Code Segments

Select the most important code segments (typically 3-5), including:
- Core algorithm implementations
- Key business logic
- Complex data processing
- Confusing edge cases

#### 5.2 Line-by-Line Analysis Format

For each key code segment, use the following format:

```markdown
### Code Segment: [Function Name]

**Purpose:** [One sentence describing what this code does]

**Original Code:**
\`\`\`python
def authenticate_user(username, password):
    user = db.find_user(username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return generate_token(user.id)
    return None
\`\`\`

**Line-by-Line Analysis:**

\`\`\`python
def authenticate_user(username, password):
    # Define user authentication function, accepting username and password as parameters
    
    user = db.find_user(username)
    # Query user from database
    # db.find_user() returns User object or None
    
    if not user:
        return None
    # When user doesn't exist, return None to indicate authentication failure
    
    if verify_password(password, user.password_hash):
        # Verify if password matches
        # verify_password() uses bcrypt to compare plaintext password with hash
        
        return generate_token(user.id)
        # Password correct, generate JWT token and return
        # generate_token() creates encrypted token containing user.id
        
    return None
    # Password incorrect, return None to indicate authentication failure
\`\`\`

**Execution Flow Example:**

Input: `username="alice", password="secret123"`

1. `db.find_user("alice")` → Returns User(id=1, username="alice", password_hash="$2b$...")
2. `user` exists, continue execution
3. `verify_password("secret123", "$2b$...")` → Returns True
4. `generate_token(1)` → Returns "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
5. Function returns token string

**Key Points:**
- Passwords are never stored in plaintext; bcrypt hashing is used
- Authentication failures always return None without exposing failure reason (security consideration)
- Token contains user ID for subsequent request verification
```

### 6. Dependencies

Explain the code's external dependencies.

```markdown
## Dependencies

### External Libraries
- **express** (v4.18.0) — Web framework
- **bcrypt** (v5.1.0) — Password hashing
- **jsonwebtoken** (v9.0.0) — JWT token generation and verification

### Internal Modules
- `database.js` — Database connection and queries
- `utils/crypto.js` — Encryption utility functions
- `middleware/auth.js` — Authentication middleware
```

### 7. Usage Examples

Provide complete, runnable examples.

```markdown
## Usage Examples

### Example 1: User Login

\`\`\`javascript
// Import module
const auth = require('./auth');

// Call authentication function
const token = await auth.authenticate_user('alice', 'secret123');

if (token) {
    console.log('Login successful! Token:', token);
    // Output: Login successful! Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
} else {
    console.log('Invalid username or password');
}
\`\`\`

**Execution Result:**
- If authentication succeeds, returns JWT token string
- If it fails, returns `null`

### Example 2: Verify Token

\`\`\`javascript
const userId = auth.verify_token(token);

if (userId) {
    console.log('Token valid, User ID:', userId);
    // Output: Token valid, User ID: 1
} else {
    console.log('Token invalid or expired');
}
\`\`\`
```

### 8. Issues & Improvements

**Optional Section** — Only include when code has obvious issues.

```markdown
## Issues & Improvements

### Performance Issues
- **Issue:** Database queried on every request, potential performance bottleneck
- **Suggestion:** Add Redis caching layer to cache user information

### Security Issues
- **Issue:** Token has no expiration time
- **Suggestion:** Add `expiresIn: '1h'` parameter in `generate_token()`

### Code Quality
- **Issue:** Lack of input validation
- **Suggestion:** Add username and password format validation
```

## Output Format

Analysis results must be output as **Markdown documents** with clear structure and easy readability.

### Standard Output Template

```markdown
# [Code Name] Source Code Analysis

## 1. Code Overview
- Programming Language:
- File Type:
- Code Scale:
- Main Dependencies:

## 2. Context & Purpose
- Business Context:
- Technical Context:
- Core Purpose:
- Use Cases:

## 3. Architecture Design
- File/Module Structure:
- Core Components:
- Data Flow:
- Design Patterns:

## 4. Core Principles
- Core Algorithms:
- Key Technologies:
- Implementation Strategy:

## 5. Detailed Implementation
### [Code Segment 1]
- Purpose:
- Original Code:
- Line-by-Line Analysis:
- Execution Flow Example:
- Key Points:

### [Code Segment 2]
...

## 6. Dependencies
- External Libraries:
- Internal Modules:

## 7. Usage Examples
- Example 1:
- Example 2:

## 8. Issues & Improvements (Optional)
```

## Quality Standards

Each code analysis must satisfy:

### Completeness
- ✅ Covers all 8 core sections (Section 8 is optional)
- ✅ Key code segments have detailed line-by-line analysis
- ✅ Algorithms and design patterns are clearly annotated with references

### Accuracy
- ✅ Technical terminology is accurate
- ✅ Code analysis is correct with no misleading content
- ✅ Examples are runnable with genuine output results

### Readability
- ✅ Language is accessible, avoiding excessive jargon
- ✅ Structure is clear using headers and lists to organize content
- ✅ Important concepts are **bolded**

### Practicality
- ✅ Execution flow examples are specific and helpful for understanding
- ✅ Usage examples are complete and runnable
- ✅ Key points highlight important aspects, avoiding redundancy

## Special Case Handling

### Multi-File Projects
1. First analyze overall project structure
2. Identify entry files and core modules
3. Analyze modules separately, explaining inter-module relationships
4. Provide module call flow diagrams (using Markdown tables or text descriptions)

### Complex Algorithms
1. First describe algorithm approach in natural language
2. Provide algorithm time/space complexity
3. Break down algorithm steps incrementally
4. Demonstrate execution process with concrete data
5. Attach authoritative reference links

### Unfamiliar Tech Stacks
1. First query and explain tech stack background
2. Explain key concepts and terminology
3. Focus on code logic rather than framework details
4. Provide learning resources for that tech stack

## Pre-Analysis Checklist

Before starting analysis, confirm:
- [ ] User's specific needs are understood (understand what? solve what problem?)
- [ ] Code's programming language and framework are identified
- [ ] Analysis scope is determined (single function vs entire project)
- [ ] Relevant reference material links are prepared
