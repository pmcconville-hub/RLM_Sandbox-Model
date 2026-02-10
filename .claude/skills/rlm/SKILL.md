---
name: rlm
description: Run a Recursive Language Model workflow for long-context tasks. Supports two modes -- automatic (RLM.completion via Modal sandbox) and manual (step-by-step with helpers and rlm-subcall subagent).
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
---

# rlm (Recursive Language Model workflow)

Use this Skill when:
- The user provides (or references) a very large context file (docs, logs, transcripts, Airtable exports, scraped webpages) that won't fit comfortably in chat context.
- You need to iteratively inspect, search, chunk, and extract information from that context.
- You can delegate chunk-level analysis to a subagent or to the RLM's built-in sub-LLM loop.

## Mental model

- Main Claude Code conversation = the root LM (orchestrator).
- RLM library (`rlm/core/rlm.py`) = the recursive engine with sandbox + sub-LLM loop.
- Modal sandbox = the isolated execution environment (safe for production data).
- Subagent `rlm-subcall` = lightweight Haiku sub-LM for manual chunk analysis.
- Helpers script (`rlm_helpers.py`) = local exploration tools (peek, grep, chunk).

## Paths

```
VENV_PYTHON=/home/pmcconville/projects/modal-sandbox/RLM_Sandbox-Model/.venv/bin/python
REPO=/home/pmcconville/projects/modal-sandbox/RLM_Sandbox-Model
HELPERS=$REPO/.claude/skills/rlm/scripts/rlm_helpers.py
```

## Inputs

This Skill reads `$ARGUMENTS`. Accept these patterns:
- `context=<path>` (required): path to the file containing the large context.
- `query=<question>` (required): what the user wants.
- `mode=auto|manual` (optional, default `manual`): which workflow to use.

If the user didn't supply arguments, ask for:
1) the context file path, and
2) the query.

---

## Mode 1: Manual (default)

Use this for exploratory work, PRP analysis, or when you want fine-grained control.

### Step 1: Initialise helpers with context

```bash
$VENV_PYTHON $HELPERS init <context_path>
$VENV_PYTHON $HELPERS status
```

### Step 2: Scout the context

```bash
$VENV_PYTHON $HELPERS exec -c "print(peek(0, 3000))"
$VENV_PYTHON $HELPERS exec -c "print(peek(len(content)-3000, len(content)))"
```

### Step 3: Choose a chunking strategy

- Prefer semantic chunking if the format is clear (markdown headings, JSON objects, log timestamps).
- Otherwise, chunk by characters (size ~200000, optional overlap).

### Step 4: Materialise chunks as files

```bash
$VENV_PYTHON $HELPERS exec <<'PY'
paths = write_chunks('.claude/rlm_state/chunks', size=200000, overlap=0)
print(len(paths))
print(paths[:5])
PY
```

### Step 5: Subcall loop (delegate to rlm-subcall)

For each chunk file, invoke the `rlm-subcall` subagent with:
- the user query,
- the chunk file path,
- and any specific extraction instructions.

Keep subagent outputs compact and structured (JSON preferred).
Append each subagent result to buffers:

```bash
$VENV_PYTHON $HELPERS exec -c "add_buffer('<paste subagent result here>')"
```

### Step 6: Synthesis

Once enough evidence is collected, synthesise the final answer in the main conversation.
Optionally export buffers for review:

```bash
$VENV_PYTHON $HELPERS export-buffers .claude/rlm_state/results.txt
```

---

## Mode 2: Automatic (Modal sandbox)

Use this for production data processing (Airtable exports, client data) where sandbox isolation is required.

### Step 1: Write a runner script

```bash
cat > /tmp/rlm_run.py <<'PYTHON'
import json
from rlm import RLM

# Load context
with open("<context_path>", "r") as f:
    context = f.read()

# Configure RLM with Modal sandbox
rlm = RLM(
    backend="anthropic",
    backend_kwargs={"model_name": "claude-sonnet-4-20250514"},
    environment="modal",
    environment_kwargs={"app_name": "rlm-svcstack", "timeout": 600},
    max_iterations=15,
    verbose=True,
)

# Run completion
result = rlm.completion(context, root_prompt="<user_query>")
print("=== RESULT ===")
print(result.response)
print("=== USAGE ===")
print(json.dumps(result.usage_summary.to_dict(), indent=2))
PYTHON
```

### Step 2: Execute

```bash
$VENV_PYTHON /tmp/rlm_run.py
```

### Step 3: Review result

The RLM will automatically:
- Load context into Modal sandbox
- Run iterative code generation + execution
- Route sub-LLM calls through the broker tunnel
- Stop when it finds a final answer or hits max_iterations
- Report token usage

---

## Guardrails

- Do not paste large raw chunks into the main chat context.
- Use helpers to locate exact excerpts; quote only what you need.
- Subagents cannot spawn other subagents. Orchestration stays in the main conversation.
- Keep scratch/state files under `.claude/rlm_state/`.
- For production data (Airtable exports, client info), always use Mode 2 (automatic/Modal).
- For exploration and dev work (PRP files, docs), Mode 1 (manual) is fine.
