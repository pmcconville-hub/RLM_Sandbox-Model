# CLAUDE.md

## Project: RLM_Sandbox-Model

Recursive Language Model framework for processing large contexts via sandboxed REPL environments and recursive sub-LLM calls. Forked from alexzhang13/rlm (MIT CSAIL research).

## Owner

Paul McConville -- SVCSTACK platform. Experiences brain fog from cancer treatment. Keep instructions chunked, confirm before irreversible actions.

## Architecture

```
RLM.completion(prompt)
    |
    v
Root LLM (Anthropic/Gemini/OpenAI)
    |
    v
Iteration loop (max 30 turns)
    |-- Generates code blocks
    |-- Executes in sandbox environment
    |-- Sub-LLM calls via broker tunnel
    |-- Checks for FINAL_VAR / final answer
    v
Result (RLMChatCompletion)
```

### Environments

| Type | Class | Use Case |
|------|-------|----------|
| `modal` | `ModalREPL` | **Primary.** gVisor sandbox on Modal.com. Safe for production data. |
| `local` | `LocalREPL` | Dev-time only. exec() on host. Never use with real data. |
| `docker` | `DockerREPL` | Local container isolation. |
| `daytona` | `DaytonaREPL` | Cloud dev environment. |
| `prime` | `PrimeREPL` | Dedicated compute. |

### LLM Backends

9 backends available: `anthropic`, `gemini`, `openai`, `azure_openai`, `portkey`, `openrouter`, `litellm`, `vllm`, `vercel`.

For SVCSTACK, use: Anthropic Sonnet (root) + Gemini Flash (sub-LLM, 90% of calls).

**Important:** Haiku does NOT work as the RLM root model. It generates XML tool-use
tags instead of the required ` ```repl ``` ` code blocks. Use Sonnet or Opus as root.
Haiku is fine as a sub-LLM (called via `llm_query` inside the sandbox).

## Setup

```bash
# Virtual environment (already created)
source /home/pmcconville/projects/modal-sandbox/RLM_Sandbox-Model/.venv/bin/activate

# Or invoke directly
/home/pmcconville/projects/modal-sandbox/RLM_Sandbox-Model/.venv/bin/python

# Modal is authenticated to workspace: paulmcconville
```

### Environment Variables

Create a `.env` file in the repo root with API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## RLM Skill & Subagent

This repo includes Claude Code integration:

- **Skill:** `/rlm` -- Invoke via `.claude/skills/rlm/SKILL.md`
- **Subagent:** `rlm-subcall` -- Haiku-based chunk analyser at `.claude/agents/rlm-subcall.md`
- **Helpers:** Local REPL helpers at `.claude/skills/rlm/scripts/rlm_helpers.py`

## Key Files

| File | Purpose |
|------|---------|
| `rlm/core/rlm.py` | Main RLM class with `completion()` entry point |
| `rlm/environments/modal_repl.py` | Modal sandbox with Flask broker pattern |
| `rlm/clients/__init__.py` | Backend router (9 providers) |
| `rlm/utils/prompts.py` | System prompts for RLM reasoning loop |
| `modal_sandbox_runner.py` | Test runner (`--test`, `--demo`) |

## SVCSTACK Integration

When processing SVCSTACK data (Airtable exports, PRP files, SEO content):

1. Always use `environment="modal"` for production data
2. Use `environment="local"` only for code development and testing
3. Follow the SVCSTACK workflow protocol: read PRP, confirm plan, execute in small chunks
4. Budget: $30/month Modal compute. Monitor with `UsageSummary`.

## Safety Rules

- Never run `environment="local"` with client data or Airtable exports
- Never commit `.env` files or API keys
- The `.venv/` directory is gitignored
- Modal sandbox state is destroyed on cleanup -- no data residue
