---
name: code-reader-v2-en
description: Cognitive science-based source code deep understanding assistant (English improved version). Supports 3 analysis modes: Quick (5-10 min), Standard (15-20 min), Deep (30+ min). Combines elaborative interrogation, self-explanation testing, and retrieval practice to help truly understand and master code. Trigger words "deeply analyze this code," "help me understand this algorithm," "quickly analyze this code."
---

# Source Code Deep Understanding Analyzer v2.0

Professional code analysis tool based on cognitive science research, supporting three analysis depths to ensure true understanding rather than fluency illusion.

## Three Analysis Modes

**Quick Mode** - Efficient overview, 5-10 minutes
- Fast understanding of code structure and functionality
- Best for code reviews, quick browsing

**Standard Mode** - Balanced understanding ⭐ Recommended, 15-20 minutes
- Understand WHY and design rationale
- Best for learning new tech, code comprehension

**Deep Mode** - Complete mastery, 30+ minutes
- Application transfer testing + quality verification
- Best for interview prep, deep research

**Default: Standard Mode unless user specifies otherwise.**

---

## Mode Selection Guide

Before starting analysis, automatically select mode based on user intent:

| User Intent | Recommended Mode | Trigger Examples |
|-------------|-----------------|------------------|
| Quick browse/code review | Quick | "quickly analyze", "briefly look", "what does this do" |
| Learning/technical research | Standard | "deeply analyze", "help me understand", "explain principles" |
| Interview prep/complete mastery | Deep | "thoroughly analyze", "I need to master this", "interview related" |

**If user doesn't specify, default to Standard Mode.**

---

## Core Philosophy: Understanding First, Memorization Second

**Combat Fluency Illusion**

> "Reading code ≠ Writing code"  
> "Understanding explanations ≠ Independent implementation"  
> "Feeling like you get it ≠ Actually understanding it"

**Core Principles:**
- Understand WHY, not just WHAT
- Force self-explanation to verify true understanding
- Build concept connections, not isolated memories
- Test transfer ability through application variants

**Research Foundation:**
- [Dunlosky et al.](https://www.aft.org/ae/fall2013/dunlosky) - Elaborative interrogation significantly outperforms passive reading
- [Chi et al.](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1803_3) - Self-explainers achieve correct mental models with higher probability
- [Karpicke & Roediger](https://science.sciencemag.org/content/319/5865/966) - Retrieval practice outperforms re-reading by 250%

---

## Pre-Analysis Mandatory Check: Understanding Verification Gate

**Based on selected mode, execute corresponding verification:**

### Quick Mode - Simplified Check
- Quickly identify code type and core functionality
- List key concepts (no deep verification)

### Standard Mode - Standard Verification
- Self-explanation test for core concepts
- Verify ability to articulate WHY

### Deep Mode - Complete Verification
- Full self-explanation test
- Application transfer ability verification

**Output Format (at beginning of analysis document):**

```markdown
## Understanding Verification Status [Standard/Deep Mode Only]

| Core Concept | Self-Explanation | Understanding WHY | Application Transfer | Status |
|--------------|------------------|-------------------|---------------------|---------|
| User Authentication Flow | ✅ | ✅ | ✅ | Understood |
| JWT Token Mechanism | ✅ | ⚠️ | ❌ | ⚠️ Needs Deeper Understanding |
| Password Hashing | ✅ | ✅ | ⚠️ | Basic Understanding |
```

---

## Three Mode Output Structures

### Quick Mode Output Structure (5-10 min)

```markdown
# [Code Name] Quick Analysis

## 1. Quick Overview
- Programming language and version
- Code scale and type
- Core dependencies

## 2. Functionality Description
- What it does (WHAT)
- Brief explanation of WHY needed

## 3. Core Algorithm/Design
- Algorithm complexity (if applicable)
- Design patterns used (if applicable)
- WHY this algorithm/pattern

## 4. Key Code Segments
- 3-5 core code segments
- Brief purpose description for each

## 5. Dependencies
- External library list and purposes

## 6. Quick Usage Example
- Simple runnable example
```

### Standard Mode Output Structure (15-20 min) ⭐ Recommended

```markdown
# [Code Name] Deep Understanding Analysis

## Understanding Verification Status
[Self-explanation test results table]

## 1. Quick Overview
- Language, scale, dependencies

## 2. Background & Motivation (Elaborative Interrogation)
- WHY this code is needed
- WHY this approach
- WHY not other alternatives

## 3. Core Concepts
- List key concepts
- Each concept answers 2-3 WHYs

## 4. Algorithm & Theory
- Complexity analysis
- WHY this algorithm
- References

## 5. Design Patterns
- Patterns identified
- WHY used

## 6. Key Code Deep Analysis
- Line-by-line WHY analysis
- Execution flow examples

## 7. Dependencies & Usage Examples
- Detailed WHY comments
```

### Deep Mode Output Structure (30+ min)

```markdown
# [Code Name] Complete Mastery Analysis

[All Standard Mode content, plus:]

## 3+. Concept Network Diagram
- Core concept inventory (3 WHYs each)
- Concept relationship matrix
- Connections to existing knowledge

## 6+. Complete Execution Examples
- Multi-scenario execution flows
- Boundary conditions
- Error-prone points

## 8. Application Transfer Scenarios (at least 2)
- Scenario 1: Constant principles + modifications + WHY
- Scenario 2: Constant principles + modifications + WHY
- Extract universal patterns

## 9. Quality Verification Checklist
- Understanding depth verification
- Technical accuracy verification
- Practicality verification
- Final "four abilities" test
```

---

## Analysis Workflow (Research-Driven)

### Step 1: Quick Overview

**Goal:** Build overall mental model

**Must identify:**
- Programming language and version
- File/project scale
- Core dependencies
- Code type (algorithm, business logic, framework code, etc.)

---

### Step 2: Elaborative Interrogation - Background & Motivation

**Core Questions (Must Answer):**

1. **WHY is this code needed?**
   - What real problem does it solve?
   - What happens without this code?

2. **WHY choose this technical approach?**
   - What are alternative solutions?
   - WHY not choose other solutions?
   - What are the trade-offs?

3. **WHY is it needed at this time/scenario?**
   - In which business process is it used?
   - What are the preconditions and postconditions?

**Output Format:**

```markdown
## Background & Motivation Analysis

### Problem Essence
**Problem to Solve:** [One sentence description]

**WHY it needs solving:** [Consequences if not solved]

### Solution Choice
**Chosen Solution:** [Current implementation approach]

**WHY choose this solution:**
- Advantages: [List 2-3 key advantages]
- Disadvantages: [List 1-2 known limitations]
- Trade-offs: [Explain what was traded off]

**Alternative Solutions Comparison:**
- Solution A: [Brief description] - WHY not chosen: [Reason]
- Solution B: [Brief description] - WHY not chosen: [Reason]

### Application Scenarios
**Applicable Scenarios:** [Specific scenario description]

**WHY applicable:** [Explain why this scenario is suitable]

**Inapplicable Scenarios:** [List boundary conditions]

**WHY inapplicable:** [Explain why certain scenarios are unsuitable]
```

---

### Step 3: Concept Network Construction

**Goal:** Build connections between concepts, not isolated memories

**Must include:**

1. **Core Concept Extraction**
   - Identify all key concepts (classes, functions, algorithms, data structures)
   - Each concept must answer 3 WHYs

2. **Concept Relationship Mapping**
   - Dependency: A depends on B - WHY?
   - Comparison: A vs B - WHY choose A?
   - Composition: A + B → C - WHY combine this way?

3. **Knowledge Connections**
   - Connect to known concepts
   - Connect to design patterns
   - Connect to theoretical foundations

**Output Format:**

```markdown
## Concept Network Diagram

### Core Concept Inventory

**Concept 1: User Authentication**
- **What it is:** Process of verifying user identity
- **WHY needed:** Protect system resources from unauthorized access
- **WHY implemented this way:** Use JWT for stateless authentication, reducing server load
- **WHY not other approaches:** Session requires server storage, not conducive to horizontal scaling

**Concept 2: Password Hashing**
- **What it is:** Converting plaintext passwords to irreversible hash values
- **WHY needed:** Even if database is leaked, attackers cannot obtain original passwords
- **WHY use bcrypt:** Built-in salt, adjustable computation cost to resist brute force
- **WHY not MD5/SHA1:** Computation too fast, easily brute-forced

### Concept Relationship Matrix

| Relationship Type | Concept A | Concept B | WHY Related This Way |
|-------------------|-----------|-----------|---------------------|
| Dependency | User Auth | Password Hash | Auth process needs password verification, must hash before comparison |
| Sequence | Password Hash | Token Generation | Token generated only after password verification passes |
| Comparison | JWT | Session | JWT stateless, suitable for distributed; Session stateful, more server pressure |

### Connections to Existing Knowledge

- **Design Patterns:** [Detailed below]
- **Algorithm Theory:** [Detailed below]
- **Security Principles:** Principle of least privilege, defense in depth
```

---

### Step 4: Algorithm & Theory Deep Analysis

**Mandatory Requirements:** All algorithms and core theories must:
1. Annotate time/space complexity
2. Explain "WHY this complexity is acceptable"
3. Provide authoritative references
4. Explain when it degrades

**Output Format:**

```markdown
## Algorithm & Theory Analysis

### Algorithm: Quick Sort

**Basic Information:**
- **Time Complexity:** Average O(n log n), Worst O(n²)
- **Space Complexity:** O(log n)

**Elaborative Interrogation:**

**WHY choose Quick Sort?**
- Excellent average performance, typically fastest in practice
- In-place sorting, high space efficiency
- Cache-friendly, good locality of reference

**WHY is worst O(n²) acceptable?**
- Worst case probability extremely low (avoidable through randomization)
- Real data typically not perfectly sorted/reversed
- Can optimize with median-of-three

**WHY not other sorting algorithms?**
- Merge Sort: Requires O(n) extra space, unsuitable for memory-constrained scenarios
- Heap Sort: Though stable O(n log n), poor cache performance, slower than Quick Sort in practice
- Insertion Sort: Excellent for small datasets, but O(n²) unsuitable for large-scale data

**When does it degrade?**
- Input already sorted or reversed (solvable with randomization)
- Poor pivot selection (solvable with median-of-three)
- Many duplicate elements (optimizable with 3-way Quick Sort)

**References:**
- [Quick Sort - Wikipedia](https://en.wikipedia.org/wiki/Quicksort)
- [Quick Sort Analysis - Princeton](https://algs4.cs.princeton.edu/23quicksort/)
- [Why is QuickSort better than MergeSort?](https://stackoverflow.com/questions/70402/why-is-quicksort-better-than-other-sorting-algorithms-in-practice)

### Theoretical Foundation: JWT (JSON Web Token)

**WHY use JWT?**
- Stateless authentication, server doesn't need to store sessions
- Self-contained, token carries all necessary information
- Cross-domain friendly, suitable for microservices architecture

**WHY is JWT secure?**
- Uses signature to verify integrity
- Cannot be forged (unless private key is leaked)
- Can set expiration time (exp)

**WHY does JWT have limitations?**
- Cannot actively invalidate (unless maintaining blacklist, breaking stateless advantage)
- Token size larger (Base64 encoding increases size by ~33%)
- Sensitive info needs encryption, signature alone doesn't provide confidentiality

**References:**
- [JWT.io - Introduction](https://jwt.io/introduction)
- [RFC 7519 - JWT Specification](https://tools.ietf.org/html/rfc7519)
```

---

### Step 5: Design Pattern Recognition & Interrogation

**Mandatory Check:** Each design pattern used in code must:
1. Clearly annotate pattern name
2. Explain WHY use this pattern
3. Explain what happens without this pattern
4. Provide standard reference

**Output Format:**

```markdown
## Design Pattern Analysis

### Pattern 1: Singleton Pattern

**Application Location:** `DatabaseConnection` class

**WHY use Singleton?**
- Database connection overhead high, reusing single instance saves resources
- Avoids connection pool chaos, unified connection lifecycle management
- Global unique access point, convenient concurrency control

**WHY not using Singleton would be problematic?**
- Every operation creates new connection, resource exhaustion
- Multiple connection instances may cause transaction inconsistency
- Difficult to control concurrent access

**Implementation Details:**
```python
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # WHY initialize in __new__:
            # Ensure singleton before object creation, thread-safe
        return cls._instance
```

**WHY implement this way?**
- Use `__new__` instead of `__init__`: Control instance creation, not initialization
- Class variable `_instance`: Store unique instance
- Lazy Loading: Create only on first use

**Potential Issues:**
- ⚠️ Not thread-safe (multi-threading needs locking)
- ⚠️ Unit testing difficult (global state hard to isolate)
- ⚠️ Violates single responsibility principle (class manages its own instance)

**Better Alternatives:**
- Dependency Injection: More flexible, easier to test
- Module-level variable: Python modules are naturally singleton

**References:**
- [Singleton Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/singleton)
- [Singleton Pattern in Python - Real Python](https://realpython.com/factory-method-python/)
```

---

### Step 6: Line-by-Line Deep Analysis (Key Code Segments)

**Core Principle:**
- Select 3-5 most critical code segments
- Each line must explain "what it does" + "WHY it does this"
- Provide execution flow examples with concrete data
- Mark error-prone points and boundary conditions

**Output Format:**

```markdown
## Key Code Deep Analysis

### Code Segment 1: User Authentication Function

**Overall Purpose:** Verify username and password, return JWT Token or None

**WHY this function is needed:** Authentication is the first line of system security, must be reliable and efficient

**Original Code:**
```python
def authenticate_user(username, password):
    user = db.find_user(username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return generate_token(user.id)
    return None
```

**Line-by-Line Detailed Analysis:**

```python
def authenticate_user(username, password):
    # Define authentication function, accepting username and plaintext password
    # WHY accept plaintext password: Must verify on server, client-side hashing is insecure
    
    user = db.find_user(username)
    # Query user object by username
    # WHY query user first: Avoid password hashing for non-existent usernames (save computation)
    # Returns: User object (containing id, username, password_hash, etc.) or None
    
    if not user:
        return None
    # Return None immediately when user doesn't exist
    # WHY return None not exception: Authentication failure is normal business flow, not exceptional
    # WHY not distinguish "user doesn't exist" from "wrong password": Prevent username enumeration attack
    
    if verify_password(password, user.password_hash):
        # Verify if plaintext password matches database hash
        # WHY use verify_password not direct comparison: bcrypt verification includes constant-time comparison
        # verify_password internal flow:
        #   1. Extract salt from password_hash
        #   2. Hash plaintext password with same salt
        #   3. Constant-time compare two hashes (prevent timing attack)
        
        return generate_token(user.id)
        # Password correct, generate JWT Token containing user ID
        # WHY only pass user.id: Token should stay lightweight, avoid redundant data
        # WHY not pass sensitive info: JWT is Base64 encoded, not encrypted, can be decoded
        
    return None
    # Return None when password incorrect
    # WHY same return value as "user doesn't exist": Prevent attackers from distinguishing failure types
```

**Execution Flow Examples:**

**Scenario 1: Successful Authentication**
```
Input: username="alice", password="Secret123!"

1. db.find_user("alice")
   → Query database
   → Return User(id=42, username="alice", password_hash="$2b$12$KIX...")

2. user exists, continue

3. verify_password("Secret123!", "$2b$12$KIX...")
   → Extract salt: $2b$12$KIX...
   → Hash "Secret123!" with salt
   → Constant-time compare hashes
   → Return True

4. generate_token(42)
   → Create payload: {"user_id": 42, "exp": 1643723400}
   → Sign with private key
   → Return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo0Miwi..."

5. Function returns Token string

Duration: ~100ms (mainly bcrypt computation)
```

**Scenario 2: User Doesn't Exist**
```
Input: username="bob", password="anything"

1. db.find_user("bob")
   → Query database
   → Return None

2. user is None, return None immediately

Duration: ~5ms (only database query)
⚠️ Note: Much faster than successful auth, may leak user existence
Solution: Add fixed delay or fake hash computation to make durations similar
```

**Scenario 3: Wrong Password**
```
Input: username="alice", password="WrongPass"

1. db.find_user("alice")
   → Return User(id=42, ...)

2. user exists, continue

3. verify_password("WrongPass", "$2b$12$KIX...")
   → Hash "WrongPass"
   → Compare hashes
   → Return False

4. Return None

Duration: ~100ms (similar to successful auth)
✅ Benefit: Cannot determine password correctness by response time
```

**Key Takeaways:**

1. **Security Considerations:**
   - ✅ Plaintext password exists briefly in memory, immediately hashed for verification
   - ✅ Failure reasons not leaked (prevent username enumeration)
   - ✅ Constant-time comparison (prevent timing attack)
   - ⚠️ Potential issue: Faster response when user doesn't exist (needs optimization)

2. **Performance Optimization:**
   - ✅ Fast return when user doesn't exist, no wasted hash computation
   - ⚠️ But this causes timing leakage, need to balance security vs performance

3. **Error Handling:**
   - ✅ Use None for failure, clear and Pythonic
   - ⚠️ Caller must check return value, or risk misusing None

4. **Improvements Possible:**
   - Add logging for failed attempts (detect brute force)
   - Add rate limiting
   - Unify failure scenario response times
```

---

### Step 7: Application Transfer Test (Verify True Understanding)

**Goal:** Test if concepts can apply to different scenarios

**Must include:**
- At least 2 different domain application scenarios
- Explain how to adjust code for new scenarios
- Mark what principles remain constant, what needs modification

**Output Format:**

```markdown
## Application Transfer Scenarios

### Scenario 1: Apply User Authentication to API Key Validation

**Original Scenario:** Web user login authentication  
**New Scenario:** Third-party API key validation

**Unchanged Principles:**
- Core flow of verifying caller identity
- Hash storage of credentials (API keys should also be hashed)
- Access token generation mechanism

**Parts Needing Modification:**

```python
# Original: username + password
def authenticate_user(username, password):
    user = db.find_user(username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return generate_token(user.id)
    return None

# Transfer: API key
def authenticate_api_key(api_key):
    # WHY only one parameter: API key itself is identity + credential
    
    app = db.find_app_by_key_prefix(api_key[:8])
    # WHY query by prefix: Avoid full table scan, API key prefix as index
    
    if not app:
        return None
    
    if verify_api_key(api_key, app.key_hash):
        # WHY also hash: Prevent key leakage from database breach
        
        return generate_token(app.id, scope=app.permissions)
        # WHY add scope: API keys typically have different permission levels
        
    return None
```

**WHY transfer this way:**
- Preserve core security principles (hash storage, constant-time comparison)
- Adjust business logic (single parameter, permission scope)
- Optimize query performance (prefix indexing)

**Learned Universal Pattern:**
- Any "who's calling" verification scenario can use similar structure
- Core: Find entity → Verify credential → Generate token
- Variation: Credential form, query method, token content

### Scenario 2: Apply Quick Sort to Log Analysis

**Original Scenario:** Sort user list by ID  
**New Scenario:** Sort millions of log entries by timestamp

**Unchanged Principles:**
- Divide and conquer approach: Recursive problem decomposition
- Pivot selection: Key to performance
- In-place sorting: Save space

**Parts Needing Adjustment:**

```python
# Original: Simple quick sort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Transfer: Log sorting (external sort + optimization)
def quicksort_logs(log_file, output_file, memory_limit):
    # WHY external sort: Data volume exceeds memory, cannot load at once
    
    # 1. Sort chunks
    chunks = split_file_into_chunks(log_file, memory_limit)
    # WHY chunking: Each chunk can fit in memory for individual sorting
    
    for chunk in chunks:
        logs = load_chunk(chunk)
        
        # WHY use timsort instead of quicksort:
        # - Logs typically partially ordered (appended by time)
        # - timsort optimized to O(n) for partially ordered data
        # - Python's built-in sorted() is timsort
        logs.sort(key=lambda log: log.timestamp)
        
        save_sorted_chunk(chunk, logs)
    
    # 2. Merge sorted chunks
    merge_sorted_chunks(chunks, output_file)
    # WHY merge: Combine multiple sorted sequences into one
    
    return output_file
```

**WHY not use quick sort directly:**
- Data volume exceeds memory: Need external sort
- Logs partially ordered: timsort more optimal
- Need stable sort: Maintain order of logs with same timestamp

**Learned Universal Pattern:**
- Algorithm choice depends on data characteristics (scale, ordering, stability requirements)
- Basic principles transferable (divide-and-conquer, comparison), but implementation needs adjustment
- Super large data needs external algorithms (chunking + merging)
```

---

### Step 8: Dependencies & Usage Examples

(Similar to original but with WHY explanations)

```markdown
## Dependency Analysis

### External Libraries

**bcrypt (v5.1.0)**
- **Purpose:** Password Hashing
- **WHY choose bcrypt:**
  - Built-in salt, no manual management
  - Adjustable computation cost (cost factor)
  - Resist GPU/ASIC acceleration attacks
- **WHY not SHA256:** Computation too fast, easily brute-forced
- **WHY not scrypt/argon2:** bcrypt more mature, better compatibility

**jsonwebtoken (v9.0.0)**
- **Purpose:** JWT token generation and verification
- **WHY choose JWT:** Stateless authentication, suitable for distributed systems
- **WHY not Session:** Session requires server storage, not scalable

### Internal Module Dependencies

**database.js → auth.js**
- **Dependency reason:** Authentication needs to query user data
- **WHY this design:** Separate data access and business logic (single responsibility principle)

**utils/crypto.js → auth.js**
- **Dependency reason:** Authentication needs password hashing and verification
- **WHY encapsulate utility module:** Cryptographic logic complex, centralized management more secure

## Complete Usage Examples

(With detailed WHY comments)

### Example 1: Standard User Login Flow

```javascript
// 1. Import authentication module
const auth = require('./auth');

// 2. Receive user input (from login form)
const username = req.body.username;  // e.g., "alice"
const password = req.body.password;  // e.g., "Secret123!"

// WHY not hash password on client:
// - After client-side hashing, the hash itself becomes the "password"
// - Attacker with hash can login directly
// - Must hash with salt on server, client always sends plaintext

// 3. Call authentication function
const token = await auth.authenticate_user(username, password);

// 4. Respond based on result
if (token) {
    // Authentication successful
    res.json({
        success: true,
        token: token,
        // WHY return token: Client needs to carry it in subsequent requests
        message: 'Login successful'
    });
    
    // WHY set HTTP-only Cookie (optional):
    // res.cookie('auth_token', token, {
    //     httpOnly: true,    // WHY: Prevent XSS attack from reading
    //     secure: true,      // WHY: HTTPS transmission only
    //     sameSite: 'strict' // WHY: Prevent CSRF attack
    // });
} else {
    // Authentication failed (user doesn't exist or wrong password)
    
    // WHY not distinguish failure reason: Prevent username enumeration
    res.status(401).json({
        success: false,
        message: 'Invalid username or password'  // Vague error message
    });
    
    // WHY return 401 not 403:
    // 401 = Unauthenticated (need to provide credentials)
    // 403 = Authenticated but no permission
}
```

**Execution Result Analysis:**

**Success Path:**
```
Client request → Server verify → Return Token
Time: ~100ms
Token example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Failure Path:**
```
Client request → Server verify → Return 401 error
Time: ~100ms (similar to success, prevent timing attack)
```
```

---

### Step 9: Self-Assessment Checklist

**After analysis completion, mandatory verification of following items:**

```markdown
## Quality Verification Checklist

### Understanding Depth Verification

- [ ] **Each core concept answers 3 WHYs**
  - WHY this concept is needed
  - WHY implemented this way
  - WHY not other approaches

- [ ] **Self-explanation test passed**
  - [ ] Can explain each core concept without looking at code
  - [ ] Can articulate WHY not just WHAT
  - [ ] Can apply in different scenarios (transfer test)

- [ ] **Concept connections established**
  - [ ] Annotated dependency/comparison/composition relationships between concepts
  - [ ] Connected to existing knowledge (design patterns, algorithm theory)
  - [ ] Explained WHY for each connection

### Technical Accuracy Verification

- [ ] **Algorithm analysis complete**
  - [ ] Time/space complexity annotated
  - [ ] WHY choose this algorithm
  - [ ] WHY complexity is acceptable
  - [ ] Provided authoritative references

- [ ] **Design pattern recognition**
  - [ ] All patterns annotated
  - [ ] WHY use this pattern
  - [ ] What happens without it
  - [ ] Provided standard reference

- [ ] **Code analysis detailed**
  - [ ] Key code segments have line-by-line analysis
  - [ ] Each line includes "what it does" + "WHY it does this"
  - [ ] Provided execution examples with concrete data
  - [ ] Marked error-prone points and boundary conditions

### Practicality Verification

- [ ] **Application transfer test**
  - [ ] At least 2 different scenario transfer examples
  - [ ] Explained what stays constant, what needs change
  - [ ] Extracted universal patterns

- [ ] **Usage examples runnable**
  - [ ] Example code complete
  - [ ] Contains detailed WHY comments
  - [ ] Explained execution results

- [ ] **Issues & improvement suggestions**
  - [ ] Pointed out potential issues
  - [ ] WHY they are issues
  - [ ] Provided improvement solutions
  - [ ] WHY improvement solutions are better

### Final Verification Questions

**If not looking at original code, based on this analysis document:**

1. ✅ Can you understand the design rationale?
2. ✅ Can you independently implement similar functionality?
3. ✅ Can you apply to different scenarios?
4. ✅ Can you clearly explain to others?

If any answer is "no," the analysis is insufficient and needs supplementation.
```

---

## Output Format Summary

**Complete Analysis Document Structure:**

```markdown
# [Code Name] Deep Understanding Analysis

## Understanding Verification Status
[Self-explanation test results table]

## 1. Quick Overview
- Programming Language:
- Code Scale:
- Core Dependencies:

## 2. Background & Motivation Analysis (Elaborative Interrogation)
- Problem Essence (WHY needed)
- Solution Choice (WHY chosen + WHY not others)
- Application Scenarios (WHY applicable + WHY not)

## 3. Concept Network Diagram
- Core Concept Inventory (3 WHYs per concept)
- Concept Relationship Matrix
- Connections to Existing Knowledge

## 4. Algorithm & Theory Deep Analysis
- Each algorithm: Complexity + WHY chosen + WHY acceptable + References
- Each theory: WHY used + WHY effective + WHY limited

## 5. Design Pattern Analysis
- Each pattern: WHY used + WHY not using is problematic + Implementation details + References

## 6. Key Code Deep Analysis
- Each segment: Line-by-line analysis (what + WHY) + Execution examples + Key takeaways

## 7. Application Transfer Scenarios (at least 2)
- Each scenario: Unchanged principles + Parts needing modification + WHY transfer this way

## 8. Dependencies & Usage Examples
- Each dependency: WHY chosen + WHY not others
- Examples contain detailed WHY comments

## 9. Quality Verification Checklist
[Check all verification items]
```

---

## Special Scenario Handling

### Multi-File Projects

1. **Overall Architecture Analysis**
   - Project structure tree + WHY organized this way
   - Entry files + WHY start here
   - Module division + WHY divided this way

2. **Inter-Module Relationships**
   - Dependency graph + WHY depend this way
   - Data flow diagram + WHY flow this way
   - Call chain + WHY call this way

3. **Module-by-Module Analysis**
   - Each core module analyzed by standard workflow
   - Emphasize WHY relationships between modules

### Complex Algorithms

1. **Layered Explanation**
   - First describe approach in natural language
   - Then show structure with pseudocode
   - Finally line-by-line analysis of implementation

2. **WHY Throughout**
   - WHY choose this algorithm
   - WHY each step done this way
   - WHY complexity is what it is

3. **Visualization Aid**
   - Show execution process with concrete data
   - Explain WHY at each step

### Unfamiliar Tech Stacks

1. **Technical Background Explanation**
   - What this tech stack is
   - WHY this tech stack exists
   - WHY project chose it

2. **Key Concept Explanation**
   - Tech stack-specific concepts
   - WHY designed this way
   - Comparison with other tech stacks

3. **Learning Resources**
   - Official documentation links
   - WHY recommend these resources
   - Learning path suggestions

---

## Pre-Analysis Final Check

Before starting analysis, confirm:

- [ ] Understood user's true needs (learning? review? interview prep?)
- [ ] Identified code's language, framework, scale
- [ ] Determined analysis focus (comprehensive understanding vs specific aspect)
- [ ] Ready to always ask WHY
- [ ] Ready to conduct self-explanation test
- [ ] Ready to find concept connections
- [ ] Ready to think about application transfer

**Remember: Goal is not "finish reading code" but "truly understand code."**

---

## 📤 Output Requirements

**After analysis completion, you MUST generate a standalone Markdown document!**

### Document Generation Rules

1. **File Naming Format**
   - Format: `[code-name]-deep-analysis.md`
   - Examples: `jwt-auth-deep-analysis.md`, `quicksort-deep-analysis.md`

2. **Generation Method**
   - **Method 1 (Recommended)**: Use Write tool to create file
     ```
     After completing analysis, use Write tool to write complete analysis to a separate file
     ```

   - **Method 2**: Ask user for save path
     ```
     After completing analysis, ask user for desired file path, then use Write tool to generate
     ```

3. **File Content**
   - Complete analysis results (according to selected mode's output structure)
   - Use Markdown format
   - Preserve all formatting (headers, tables, code blocks, lists, etc.)

### Output Workflow Example

```
User: Deeply analyze this code

1. [Complete analysis process]

2. [Show analysis summary in conversation]

3. Use Write tool to generate complete document:
   File path: [code-name]-deep-analysis.md
   Content: [Complete analysis content]

4. Inform user: "Complete analysis saved to [file path]"
```

**Important: Don't just output analysis in conversation - you MUST generate a saveable Markdown file!**
