"""
Error taxonomy from the MAST project Codebook.xlsx.

Defines step-level error types for Multi-Agent System (MAS) trace analysis.
Three top-level origins, each with subcategories:

  MAS-level Errors  (M1 – M5)
  Tool-side Errors  (T1 – T4)
  User-side Errors  (U1 – U2)

Reference: MAST-Data/Codebook.xlsx
           MAST-Data/MAST/taxonomy_definitions_examples/definitions.txt
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Full taxonomy  –  exactly matches Codebook.xlsx rows
# ═══════════════════════════════════════════════════════════════════════════════
ERROR_TYPES = {
    # ── MAS-level Errors ───────────────────────────────────────────────────
    # M1. Task understanding and spec handling
    "M1.1": "Disobey-Task-Specification",
    "M1.2": "Forget-corner-case",
    "M1.3": "Fail-to-meet-implicit-requirements",
    # M2. Inter-agent communication and coordination
    "M2.1": "Ignore-peer-response",
    "M2.2": "Ignore-peer-review",
    "M2.3": "Coordination-issue",
    "M2.4": "Missing-usage-instructions",
    "M2.5": "Data-mishandling",
    # M3. Planning, summarization, and verification
    "M3.1": "Redundant-plan",
    "M3.2": "Wrong-summary",
    "M3.3": "Incorrect-verification",
    "M3.4": "Premature-termination",
    # M4. Internal reasoning and honesty
    "M4.1": "Hallucinated-reasoning",
    "M4.2": "Agent-hide-error",
    # M5. Implementation / infra issues
    "M5.1": "MAS-technical-issue",
    "M5.2": "Inefficient-code",
    # ── Tool-side Errors ───────────────────────────────────────────────────
    # T1. Buggy tools
    "T1.1": "Wrong-tool-implementation",
    # T2. Tool selection and coverage
    "T2.1": "Incomplete-tool-selection",
    "T2.2": "Wrong-tool-selection",
    # T3. Tool existence and schema
    "T3.1": "Use-non-existing-tool",
    "T3.2": "Use-non-existing-context",
    "T3.3": "Incomplete-context",
    # T4. Tool invocation patterns
    "T4.1": "Repetitive-tool-calling",
    "T4.2": "Wrong-tool-calling-order",
    # ── User-side Errors ───────────────────────────────────────────────────
    # U1. Task specification
    "U1.1": "Ambiguous-task-description",
    "U1.2": "Conflicting-requirements",
    # U2. Feedback
    "U2.1": "Unhelpful-feedback",
    "U2.2": "Misleading-feedback",
}

# ── Definitions (from Codebook.xlsx, used by heuristic & LLM evaluators) ──────
ERROR_DEFINITIONS = {
    "M1.1": "Ignores explicit constraints or instructions (wrong modality, wrong output format, ignores required conditions).",
    "M1.2": "Fails to handle edge cases.",
    "M1.3": "Misses domain-typical expectations not spelled out (robustness, completeness, safety).",
    "M2.1": "Fails to consider another agent's output when making decisions.",
    "M2.2": "Fails to report errors found in peers' responses.",
    "M2.3": "Generic coordination problems: conflicting plans, duplicate work, unclear role boundaries.",
    "M2.4": "Fails to communicate how generated artifacts (code, configs, data) should be used by other agents.",
    "M2.5": "Data given by one agent got mistaken after being transferred to another (approximated, replaced, incorrectly modified).",
    "M3.1": "Over-detailed or repeated planning/processing that doesn't clearly improve robustness.",
    "M3.2": "Summary misrepresents or omits crucial facts from prior content.",
    "M3.3": "Performs a check but uses wrong criteria, misinterprets results, or incorrectly declares something correct/incorrect.",
    "M3.4": "MAS ends the task while important subgoals remain incomplete (especially due to linear, non-iterative design).",
    "M4.1": "Inconsistent internal reasoning over time (switches values without justification).",
    "M4.2": "Agent detects or encounters an error but hides/suppresses it instead of surfacing for correction.",
    "M5.1": "Infrastructure bugs (Flask errors, logging issues, crashes) that affect MAS behavior but are not reasoning per se.",
    "M5.2": "Code produced is correct but inefficient (e.g., suboptimal algorithm) in ways that meaningfully affect resource usage.",
    "T1.1": "Wrong tool implementation (e.g., hard-coded tools).",
    "T2.1": "Fails to use all necessary tools for the task (e.g., never calls a checker or a complementary API).",
    "T2.2": "Chooses an inappropriate tool given the task or current subgoal.",
    "T3.1": "Calls a tool that doesn't exist or isn't available (naming errors, wrong endpoint).",
    "T3.2": "Constructs tool calls using context/variables that were never defined (referencing a file or ID that doesn't exist).",
    "T3.3": "Tool is called without all required parameters or with missing/reduced context compared to what agents had.",
    "T4.1": "Repeatedly calls the same tool with essentially identical arguments without meaningful progress (loops).",
    "T4.2": "Calls tools in a logically incorrect or inefficient sequence.",
    "U1.1": "Missing or vague requirements that reasonably confuse a MAS.",
    "U1.2": "Different parts of the user prompt / follow-ups contradict each other.",
    "U2.1": "User signals dissatisfaction without actionable guidance.",
    "U2.2": "User incorrectly confirms a wrong intermediate or final answer.",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Derived constants
# ═══════════════════════════════════════════════════════════════════════════════
ERROR_TYPE_LIST = list(ERROR_TYPES.keys())
NUM_ERROR_TYPES = len(ERROR_TYPE_LIST)           # 28
ERROR_CODE_TO_IDX = {code: idx for idx, code in enumerate(ERROR_TYPE_LIST)}
ERROR_IDX_TO_CODE = {idx: code for code, idx in ERROR_CODE_TO_IDX.items()}

# ── Subcategory groupings  ────────────────────────────────────────────────────
# MAS-level
M1_TASK_SPEC_ERRORS       = ["M1.1", "M1.2", "M1.3"]
M2_COMMUNICATION_ERRORS   = ["M2.1", "M2.2", "M2.3", "M2.4", "M2.5"]
M3_PLAN_VERIFY_ERRORS     = ["M3.1", "M3.2", "M3.3", "M3.4"]
M4_REASONING_ERRORS       = ["M4.1", "M4.2"]
M5_INFRA_ERRORS           = ["M5.1", "M5.2"]
# Tool-side
T1_BUGGY_TOOL_ERRORS      = ["T1.1"]
T2_TOOL_SELECTION_ERRORS   = ["T2.1", "T2.2"]
T3_TOOL_SCHEMA_ERRORS     = ["T3.1", "T3.2", "T3.3"]
T4_TOOL_INVOCATION_ERRORS = ["T4.1", "T4.2"]
# User-side
U1_TASK_SPEC_ERRORS       = ["U1.1", "U1.2"]
U2_FEEDBACK_ERRORS        = ["U2.1", "U2.2"]

# Higher-level groupings (for reward computation)
MAS_ERRORS  = M1_TASK_SPEC_ERRORS + M2_COMMUNICATION_ERRORS + M3_PLAN_VERIFY_ERRORS + M4_REASONING_ERRORS + M5_INFRA_ERRORS
TOOL_ERRORS = T1_BUGGY_TOOL_ERRORS + T2_TOOL_SELECTION_ERRORS + T3_TOOL_SCHEMA_ERRORS + T4_TOOL_INVOCATION_ERRORS
USER_ERRORS = U1_TASK_SPEC_ERRORS + U2_FEEDBACK_ERRORS

# Backward-compatible aliases used by error_reward.py
SPECIFICATION_ERRORS = M1_TASK_SPEC_ERRORS
COMMUNICATION_ERRORS = M2_COMMUNICATION_ERRORS
VERIFICATION_ERRORS  = M3_PLAN_VERIFY_ERRORS

# ── Severity weights ──────────────────────────────────────────────────────────
# Verification / termination / reasoning errors are most damaging to final output
# Communication / coordination errors moderately damaging
# Tool & user errors weighted lower (not directly the MAS's "fault" in routing)
DEFAULT_ERROR_WEIGHTS = {code: 1.0 for code in ERROR_TYPE_LIST}
# M3 (verification/planning) and M4 (reasoning) – high impact
for code in M3_PLAN_VERIFY_ERRORS + M4_REASONING_ERRORS:
    DEFAULT_ERROR_WEIGHTS[code] = 1.5
# M1 (task spec) – moderately high
for code in M1_TASK_SPEC_ERRORS:
    DEFAULT_ERROR_WEIGHTS[code] = 1.2
# Tool-side errors – moderate
for code in TOOL_ERRORS:
    DEFAULT_ERROR_WEIGHTS[code] = 0.8
# User-side errors – low (not attributable to agents)
for code in USER_ERRORS:
    DEFAULT_ERROR_WEIGHTS[code] = 0.3


def format_error_type(code: str) -> str:
    """Return 'M1.1 Disobey-Task-Specification' style string."""
    return f"{code} {ERROR_TYPES.get(code, 'Unknown')}"


def get_codebook_text() -> str:
    """Return a formatted text version of the full codebook for LLM prompts."""
    lines = []
    current_category = ""
    for code, name in ERROR_TYPES.items():
        # Detect category changes
        prefix = code.split(".")[0]
        if prefix[0] == "M":
            cat = "MAS-level Error"
        elif prefix[0] == "T":
            cat = "Tool-side Error"
        else:
            cat = "User-side Error"
        if cat != current_category:
            if current_category:
                lines.append("")
            lines.append(f"## {cat}")
            current_category = cat

        definition = ERROR_DEFINITIONS.get(code, "")
        lines.append(f"  {code} {name}: {definition}")
    return "\n".join(lines)
