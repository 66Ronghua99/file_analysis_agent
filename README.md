# Local File Analysis Agent

This repository contains a Python 3.10+ package and one-shot CLI for read-only
analysis of files below one absolute workspace directory. The agent can list
directories, search UTF-8 text, read bounded line ranges, and delegate one
focused question to an independent subagent. It never writes files or starts a
vLLM process.

## Install

```bash
python -m pip install -e '.[dev]'
```

Provider SDKs are optional:

```bash
python -m pip install -e '.[openai]'   # OpenRouter or external vLLM
python -m pip install -e '.[litellm]'  # LiteLLM
```

## One-shot usage

The workspace must be an existing absolute directory. The task may be a single
argument or one stdin payload.

```bash
file-analysis-agent \
  --workspace /home/me/project \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  "Find the configuration used by the build and summarize it."
```

For an externally managed vLLM endpoint:

```bash
file-analysis-agent \
  --workspace /home/me/project \
  --provider vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-model \
  "List the entry points and their modules."
```

Chat Completions is the default API mode. Use `--api-mode responses` only when
the selected model and endpoint support the Responses format. `--max-steps`,
`--context-window`, and `--prompt-dir` are available for bounded execution and
local prompt/schema overrides.

Offline smoke check:

```bash
file-analysis-agent \
  --workspace "$PWD" \
  --provider fake \
  --model fake \
  "Confirm the CLI wiring."
```

The package API accepts an injected fake client for deterministic tests:

```python
from pathlib import Path

from file_analysis_agent import Agent, AgentConfig
from file_analysis_agent.clients.fake import StaticClient

config = AgentConfig(workspace_dir=Path.cwd(), provider="fake", model="fake")
answer = Agent(config, client=StaticClient("done")).run("Check the repository.")
```

All filesystem tools resolve paths through the configured workspace boundary.
They return bounded, model-readable error strings for traversal, missing files,
permission failures, wrong types, and UTF-8 decode failures. Context compression
keeps the root prompt immutable, retains the latest five complete ReAct rounds,
and validates structured Blackboard updates before committing them. This is an
application-level boundary; it is not an OS sandbox and does not claim to stop
a separate local process from changing symlinks concurrently.
