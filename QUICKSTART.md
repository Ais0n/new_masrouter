# 🚀 Quick Start: Using MasRouter with Your Own Prompts

## Three Ways to Use MasRouter

I've created **3 demo scripts** for you to interact with MasRouter. Choose based on your needs:

---

## 1️⃣ **Minimal Example** (Easiest)
**File:** `example_minimal.py`

**Best for:** Quick testing, simple API usage

**What it does:**
- Runs 2 example queries automatically
- Lets you enter your own query
- Shows answer, cost, and number of agents used
- Clean, minimal output

**Run it:**
```bash
python example_minimal.py
```

**Example output:**
```
Query: What is 25% of 80 plus 15?
Answer: The answer is 35. [detailed reasoning...]
Cost: $0.0123
Agents Used: 2.3
```

---

## 2️⃣ **Decision Viewer** (Recommended First Step)
**File:** `demo_simple.py`

**Best for:** Understanding how the router makes decisions

**What it does:**
- Shows what the router SELECTS without executing agents
- ✅ Task classification (Math/Code/Commonsense)
- ✅ Collaboration method (CoT/Debate/etc.)
- ✅ Number of agents
- ✅ Roles assigned to each agent
- ✅ LLMs selected for each agent
- ✅ Fast (no API calls to LLMs)
- ✅ Free (no costs)

**Run it:**
```bash
python demo_simple.py
```

**Example output:**
```
🎯 ROUTER DECISIONS:
────────────────────────────────────────────────────────────────────────────────
📋 Task Domain:        Code
🤝 Collaboration:      Chain
👥 Number of Agents:   3

👤 Selected Roles:
   Agent 1: AlgorithmDesigner
   Agent 2: CodeWriter
   Agent 3: BugFixer

🧠 Selected LLMs:
   Agent 1: google/gemini-2.0-flash-001
   Agent 2: deepseek/deepseek-chat
   Agent 3: openai/gpt-4o-mini

📊 Task Classification Confidence:
   Math            [  8.5%] ████
   Commonsense     [ 12.3%] ██████
   Code            [ 79.2%] ███████████████████████████████████████
```

---

## 3️⃣ **Full Interactive Demo** (Complete Experience)
**File:** `demo_interactive.py`

**Best for:** Getting actual answers from the multi-agent system

**What it does:**
- Runs the COMPLETE multi-agent system
- Actually executes the collaboration
- Provides final answers
- ⚠️ Requires valid API keys
- ⚠️ Costs money (calls multiple LLMs)
- ⚠️ Slower (multiple API calls)

**Run it:**
```bash
python demo_interactive.py
```

**Example output:**
```
💬 Your Query: Solve: 2x + 5 = 15

🔄 Processing your query through MasRouter...

Results
════════════════════════════════════════════════════════════════════════════════
✅ Final Answer:
To solve 2x + 5 = 15:
Step 1: Subtract 5 from both sides: 2x = 10
Step 2: Divide both sides by 2: x = 5
The answer is x = 5.

💰 Cost: $0.0234
🔢 Number of Agents Used: 2.87
```

---

## ⚙️ Setup Requirements

### For Minimal Example & Decision Viewer:
```bash
# Just need the code - no API keys required!
python demo_simple.py  # Works immediately
```

### For Full Interactive Demo:
```bash
# 1. Setup environment file
cp template.env .env

# 2. Edit .env and add your credentials:
#    URL = "https://openrouter.ai/api/v1"  # or your LLM provider
#    KEY = "your-api-key-here"

# 3. Run the demo
python demo_interactive.py
```

---

## 📊 Comparison

| Feature | Minimal | Decision Viewer | Full Interactive |
|---------|---------|----------------|------------------|
| **Execution Time** | Fast | Very Fast | Slow (minutes) |
| **API Costs** | Yes ($) | No (Free) | Yes ($$) |
| **Shows Decisions** | No | ✅ Yes | ✅ Yes |
| **Gets Real Answers** | ✅ Yes | No | ✅ Yes |
| **Requires API Keys** | ✅ Yes | No | ✅ Yes |
| **Best For** | Quick tests | Understanding | Production use |

---

## 💡 Recommended Workflow

**Step 1:** Start with **Decision Viewer** (`demo_simple.py`)
- No setup needed
- See how routing works
- Understand the architecture
- Free and fast

**Step 2:** Try **Minimal Example** (`example_minimal.py`)  
- Setup API keys
- Test with simple queries
- Check if answers make sense
- Monitor costs

**Step 3:** Use **Full Interactive** (`demo_interactive.py`)
- For complex queries
- When you need detailed answers
- Production-ready interface

---

## 🎯 Example Queries to Try

### Math Problems:
```
- What is 15% of 240 plus 30?
- Solve the equation: 3x + 7 = 22
- If a train travels 60 mph for 2.5 hours, how far does it go?
```

### Coding Problems:
```
- Write a Python function to check if a number is prime
- Create a function that reverses a string without using built-in methods
- Implement a binary search algorithm
```

### Commonsense Questions:
```
- Why do birds fly south for the winter?
- What happens when you mix baking soda and vinegar?
- Why do we need sleep?
```

---

## 🔍 What Happens Behind the Scenes?

When you submit a query, MasRouter:

1. **Encodes your query** into a vector representation (384 dimensions)
2. **Classifies the task type** (Math/Code/Commonsense) using similarity matching
3. **Selects collaboration method** (CoT/Debate/Chain/etc.) via VAE encoding
4. **Determines agent count** (1-6) based on query difficulty
5. **Allocates roles** sequentially (e.g., Planner → Solver → Verifier)
6. **Routes to LLMs** probabilistically based on agent capabilities
7. **Constructs a Graph** with the selected agents and topology
8. **Executes the collaboration** following the chosen reasoning pattern
9. **Aggregates results** and returns the final answer

---

## 📚 More Information

- **Full Guide:** See `USAGE_GUIDE.md` for detailed documentation
- **Paper:** https://arxiv.org/abs/2502.11133
- **Code Structure:** Explore `MAR/` directory for implementation details

---

## 🆘 Need Help?

**Common Issues:**

1. **"No module named 'MAR'"**
   - Make sure you run from repository root directory

2. **"API Key Error"**
   - Check that `.env` file exists with valid credentials
   - Verify your API provider URL and key

3. **"CUDA out of memory"**
   - Use CPU: Scripts will auto-detect
   - Or set: `device = torch.device('cpu')`

4. **"No pre-trained model found"**
   - That's OK! Will use random initialization
   - For better results, train a model first

---

## 🎉 You're Ready!

Start with:
```bash
python demo_simple.py
```

Then try your own queries! 🚀
