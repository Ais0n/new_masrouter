# MasRouter Usage Guide

## 🎯 How to Use MasRouter as a Multi-Agent System

This guide explains how to interact with MasRouter using your own prompts and queries.

---

## 📋 Quick Start

### 1. **Setup Environment**

First, make sure you have your API keys configured:

```bash
# Copy the template and add your API keys
cp template.env .env

# Edit .env and add:
# URL = "your-llm-api-url"  # e.g., https://openrouter.ai/api/v1
# KEY = "your-api-key"
```

### 2. **Choose Your Demo**

I've created **two demo scripts** for you:

#### **Option A: Simple Decision Viewer** (Recommended to start)
Shows what MasRouter *selects* without running the full system (fast, no API costs):

```bash
python demo_simple.py
```

**Features:**
- ✅ See task classification
- ✅ See collaboration method selection
- ✅ See number of agents determined
- ✅ See roles allocated
- ✅ See LLMs selected
- ❌ Does NOT execute the actual multi-agent system (saves time & money)

#### **Option B: Full Interactive Demo**
Runs the complete multi-agent system with your queries (slower, uses API):

```bash
python demo_interactive.py
```

**Features:**
- ✅ Complete end-to-end execution
- ✅ Real multi-agent collaboration
- ✅ Final answers from the system
- ⚠️ Requires API access and will cost money

---

## 🔍 Understanding the System

### What MasRouter Does

MasRouter automatically constructs an optimal multi-agent system by making 5 key decisions:

```
Your Query
    ↓
1️⃣ Task Classification (Math/Code/Commonsense)
    ↓
2️⃣ Collaboration Method (CoT/Debate/Reflection/Chain/etc.)
    ↓
3️⃣ Number of Agents (1-6 agents)
    ↓
4️⃣ Role Allocation (Specific roles for each agent)
    ↓
5️⃣ LLM Selection (Which LLM for each agent)
    ↓
Multi-Agent System Executes
    ↓
Final Answer
```

---

## 📝 Example Usage

### Using Demo Simple (Decision Viewer)

```bash
$ python demo_simple.py

🤖 MasRouter Decision Viewer
════════════════════════════════════════════════════════════════════════════════

EXAMPLE QUERIES
════════════════════════════════════════════════════════════════════════════════

[1] Write a Python function to check if a number is prime.
[2] What is 15% of 240 plus 30?
[3] Why do birds fly south for the winter?
[4] Solve the equation: 3x + 7 = 22
[5] Create a function that reverses a string...

Enter query number (1-5) or type your own query: 1

┌──────────────────────────────────────────────────────────────────────────────┐
│ Query: Write a Python function to check if a number is prime.               │
└──────────────────────────────────────────────────────────────────────────────┘

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

### Using Demo Interactive (Full System)

```bash
$ python demo_interactive.py

🤖 Welcome to MasRouter Interactive Demo!

💬 Your Query: Calculate the sum of all prime numbers less than 100.

🔄 Processing your query through MasRouter...
   (This may take a moment as it calls multiple LLMs)

Results
════════════════════════════════════════════════════════════════════════════════
📝 Your Query: Calculate the sum of all prime numbers less than 100.

✅ Final Answer:
The sum of all prime numbers less than 100 is 1060.
[Detailed reasoning from agents would appear here...]

💰 Cost: $0.0234
🔢 Number of Agents Used: 2.87
📊 Total API Cost: $0.0234
```

---

## 🎨 Customization Options

### 1. **Add Your Own Tasks**

Edit `MAR/Prompts/tasks_profile.py`:

```python
tasks_profile = [
    {'Name': 'YourTask', 'Description': 'Description of your custom task...'},
    # ... existing tasks
]
```

### 2. **Add Your Own Collaboration Methods**

Edit `MAR/Agent/reasoning_profile.py`:

```python
reasoning_profile = [
    {'Name': 'YourMethod', 'Description': 'How your method works...'},
    # ... existing methods
]
```

### 3. **Add Your Own LLMs**

Edit `MAR/LLM/llm_profile.py`:

```python
llm_profile = [
    {'Name': 'your-provider/your-model', 
     'Description': 'Model description with benchmarks...'},
    # ... existing LLMs
]
```

### 4. **Add Custom Roles**

Add JSON files in `MAR/Roles/{TaskName}/`:

```json
{
  "Name": "YourRole",
  "Description": "What this role does...",
  "Prompt": "System prompt for this role..."
}
```

---

## 🔧 Advanced Usage

### Direct API Usage

```python
import torch
from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile

# Initialize router
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
router = MasRouter(max_agent=6, device=device).to(device)

# Load pre-trained weights
router.load_state_dict(torch.load("mbpp_router_epoch0_new.pth", map_location=device))
router.eval()

# Run inference
with torch.no_grad():
    results, costs, log_probs, tasks_probs, vae_loss, agent_num = router.forward(
        queries=["Your query here"],
        tasks=tasks_profile,
        llms=llm_profile,
        collabs=reasoning_profile,
        given_task=None,  # Auto-detect, or specify index
        prompt_file='MAR/Roles/FinalNode/gsm8k.json'
    )

print(f"Answer: {results[0]}")
print(f"Cost: ${costs[0]:.4f}")
```

### Batch Processing

```python
# Process multiple queries at once
queries = [
    "Query 1...",
    "Query 2...",
    "Query 3..."
]

results, costs, log_probs, tasks_probs, vae_loss, agent_num = router.forward(
    queries=queries,
    tasks=tasks_profile,
    llms=llm_profile,
    collabs=reasoning_profile
)

for query, result, cost in zip(queries, results, costs):
    print(f"Q: {query}")
    print(f"A: {result}")
    print(f"Cost: ${cost:.4f}\n")
```

---

## 🎓 Training Your Own Router

To train the router on your own dataset:

```bash
# For code tasks (MBPP)
python Experiments/run_mbpp.py --epochs 10 --batch_size 16 --lr 0.01

# For math tasks (GSM8K)
python Experiments/run_gsm8k.py --epochs 10 --batch_size 16 --lr 0.01

# For general tasks (MMLU)
python Experiments/run_mmlu.py --epochs 10 --batch_size 16 --lr 0.01
```

---

## 💡 Tips & Best Practices

1. **Start with `demo_simple.py`** to understand routing decisions without API costs
2. **Use `demo_interactive.py`** when you want actual answers from the multi-agent system
3. **Pre-trained models** work best for their trained domain (e.g., `mbpp_router` for code)
4. **Monitor costs** - each query can use multiple LLM calls
5. **Experiment with max_agent** parameter to control complexity vs. cost tradeoff

---

## 🐛 Troubleshooting

### "No pre-trained model found"
- Download or train a model first using the training scripts
- Or continue with random initialization (won't be optimal)

### "API Key Error"
- Make sure `.env` file exists with valid URL and KEY
- Check that your API provider supports the models in `llm_profile`

### "CUDA out of memory"
- Use CPU: Set `device = torch.device('cpu')`
- Reduce batch_size in training scripts
- Use smaller models

### "Import Error"
- Make sure you're running from the repository root directory
- Check that all dependencies are installed: `pip install -r requirements.txt` (if exists)

---

## 📚 Understanding the Components

### Key Modules:

1. **TaskClassifier**: Determines if query is Math/Code/Commonsense
2. **CollabDeterminer**: Selects collaboration strategy (CoT/Debate/etc.)
3. **NumDeterminer**: Decides how many agents are needed (1-6)
4. **RoleAllocation**: Assigns specific roles to each agent
5. **LLMRouter**: Selects which LLM to use for each agent

### Architecture:
```
Text Encoder (SentenceTransformer)
        ↓
    [VAE Encoders] ← Learn latent representations
        ↓
  [Task Classifier] → Softmax selection
        ↓
[Collab Determiner] → Stochastic sampling
        ↓
  [Num Determiner] → Regression + rounding
        ↓
 [Role Allocation] → Sequential selection
        ↓
   [LLM Router] → Multinomial allocation
        ↓
    Graph Construction & Execution
```

---

## 🎯 Next Steps

1. **Try the demos** with your own queries
2. **Explore the codebase** to understand the architecture
3. **Train on your own data** for custom domains
4. **Contribute** improvements or new features

For more details, see the [paper](https://arxiv.org/abs/2502.11133) or explore the code!
