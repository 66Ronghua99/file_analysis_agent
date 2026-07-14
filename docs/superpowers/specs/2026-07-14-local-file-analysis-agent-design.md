# Local File Analysis Agent Design

- Date: 2026-07-14
- Status: Approved for implementation planning
- Scope: A reusable Python package and a one-shot CLI for safe, local, read-only file analysis.

## 1. Goals and non-goals

The first version will provide:

- application-level path isolation under an absolute `workspace_dir`;
- safe directory listing, recursive text search, and paginated file reading;
- a provider-independent synchronous ReAct loop;
- OpenRouter, external vLLM, and LiteLLM client adapters;
- independent subagent execution with its own loop, system prompt, messages, and Blackboard;
- short-term context management using a per-agent Blackboard state;
- a Python package API and a one-shot CLI.

The first version will not provide:

- file writing, editing, or deletion;
- an interactive multi-turn CLI;
- persistent memory across CLI invocations;
- operating-system or container-level sandboxing;
- management of the vLLM process;
- a global multi-agent coordinator.

## 2. Package boundaries

The implementation should use focused modules with dependency injection at the boundaries:

```text
file_analysis_agent/
├── agent/
│   ├── loop.py              # ReAct execution loop
│   ├── subagent.py          # Independent subagent construction
│   └── models.py            # Internal request, response, and state models
├── clients/
│   ├── protocol.py          # LLMClient protocol and normalized responses
│   ├── openai_compatible.py # OpenRouter and vLLM
│   └── litellm_client.py    # LiteLLM
├── sandbox/
│   ├── resolver.py          # Workspace path resolution and validation
│   └── errors.py            # Tool-facing error values
├── tools/
│   ├── filesystem.py        # list_dir, search_text, read_file
│   └── registry.py          # Schema loading, validation, and dispatch
├── memory/
│   ├── blackboard.py        # Per-agent Blackboard state
│   └── token_counter.py     # Injectable token counting
├── resources/
│   ├── prompts/
│   │   ├── system.md
│   │   ├── subagent_system.md
│   │   └── blackboard_update.md
│   └── tools/
│       └── file_tools.json
├── config.py
└── cli.py
```

`AgentLoop` knows only the `LLMClient`, Blackboard memory, and `ToolRegistry` protocols. It must not contain provider-specific parsing or direct filesystem access. Prompt text and Function Calling schemas are package resources and are never embedded in the loop.

The minimal public API is `Agent`, `AgentConfig`, `LLMClient`, `Sandbox`, `ToolRegistry`, and `BlackboardState`.

## 3. Sandbox and read-only file tools

`workspace_dir` must be an absolute existing directory. Initialization resolves it with `strict=True` and rejects any other kind of root.

Every user-supplied path is handled by one resolver:

1. relative paths are interpreted relative to the workspace root;
2. absolute paths are allowed only if their resolved target is inside the root;
3. `Path.resolve()` removes `..` components and follows Symlinks;
4. `resolved.is_relative_to(workspace_root)` is the final boundary check.

An out-of-bounds path returns exactly:

```text
[Error] Permission Denied: out of bounds
```

The sandbox is an application-level path boundary. It does not claim to defend against an untrusted local process changing filesystem links concurrently; OS-level isolation is outside this version's scope.

### `list_dir(path: str) -> str`

Lists direct children of a validated directory in stable sorted order. Missing paths, permission failures, and type errors are returned as tool-facing error strings.

### `search_text(keyword: str, path: str) -> str`

Recursively searches UTF-8 text files below a validated directory. Each match includes the workspace-relative path, 1-based line number, and matching line. Results are bounded by a configurable maximum (default 200 matches) and include an explicit truncation notice when the limit is reached. Directory Symlinks are resolved before traversal; targets outside the workspace are skipped and reported.

### `read_file(filepath: str, start_line: int = 1, end_line: int | None = None) -> str`

Reads a validated UTF-8 file with 1-based line numbers. A single call returns at most 300 lines. A larger requested range is clamped and accompanied by an explicit truncation notice. Invalid ranges and decode failures return clear errors.

Known operational failures are converted to stable strings such as:

```text
[Error] File Not Found: ...
[Error] Permission Denied: ...
[Error] Invalid Path: ...
[Error] Not A Directory: ...
[Error] Is A Directory: ...
[Error] Decode Failed: ...
```

They do not escape the tool boundary as uncaught exceptions.

## 4. LLM client abstraction

The Agent depends on a synchronous protocol equivalent to:

```text
complete(request: CompletionRequest) -> LLMResponse
```

`CompletionRequest` contains normalized messages, tool definitions, model settings, and the selected API mode. `LLMResponse` contains assistant text, normalized tool calls, finish reason, usage, and an optional raw provider response for diagnostics.

`OpenAICompatibleClient` serves OpenRouter and an externally managed vLLM HTTP endpoint. `LiteLLMClient` calls LiteLLM directly. The provider choice, model, base URL, credential source, and API mode are configuration values and never branch the Agent loop.

Chat Completions is the default mode. Responses-format support is implemented in adapters and normalized to the same internal response model. An adapter reports an explicit error when its selected provider or model does not support the requested mode; it does not silently switch modes.

OpenRouter/vLLM and LiteLLM dependencies should be optional package extras. The core package must remain usable with a fake client for tests without network access. No LangChain dependency is required.

## 5. Blackboard memory

Each Agent owns one mutable, in-process `BlackboardState`:

```text
task
facts
findings
decisions
open_questions
progress
next_action
```

Facts and findings retain provenance where available, especially file paths and line ranges. The original task is stored in `task` and remains available after message compaction. Main Agent and subagent Blackboards are independent.

The root `system.md` content is immutable. Every model request is assembled as:

```text
system.md                  # unchanged root prompt
blackboard system packet   # serialized current BlackboardState
latest conversation data
```

The latest conversation data contains the newest five complete ReAct rounds. A round is one LLM decision followed by its tool calls and observations. The active, unfinished round is never removed.

An injected `TokenCounter` estimates the assembled request. When the context reaches 80% of the configured context window:

1. the latest five complete rounds are retained;
2. older rounds and the existing Blackboard are passed to `BlackboardUpdater`;
3. the updater uses `blackboard_update.md`, no tools, and returns validated structured JSON;
4. the new Blackboard is committed atomically only after validation succeeds;
5. older raw rounds are removed and the Blackboard packet is injected on the next request.

The Blackboard packet has its own configurable budget, defaulting to approximately 20% of the context window. The updater merges duplicate entries and removes obsolete information so compression does not merely move context rot from messages into the Blackboard. If updating fails, the old state and old messages remain intact and the Agent terminates with an explicit compression error.

The v1 default is a deterministic `ApproximateTokenCounter` based on the UTF-8 size of the serialized request plus a fixed per-message overhead. It is intentionally conservative and is replaceable through the `TokenCounter` protocol when a provider-specific tokenizer is available. The context window is explicit configuration rather than automatic provider discovery. If the serialized Blackboard plus the latest five rounds cannot fit within the configured context window after a valid update, the Agent returns an explicit context-budget error rather than dropping the active round or silently truncating the Blackboard.

## 6. ReAct loop and subagents

`AgentLoop` initializes the root prompt, Blackboard, and user task, then repeats up to `max_steps=100` LLM decision rounds:

- check and compact context;
- call the normalized client;
- finish on a final assistant response without tool calls;
- otherwise validate and execute tool calls sequentially;
- append each tool result as an Observation and continue.

Unknown tools, malformed arguments, and filesystem failures become Observations so the model can correct itself. Provider configuration/network failures and Blackboard update failures terminate clearly instead of entering an uncontrolled retry loop.

`call_subagent` constructs a fresh AgentLoop with `subagent_system.md`, a new Blackboard, and a new message history. It receives only the caller's prompt, uses the same client configuration, and has a registry that excludes `call_subagent`. Its final text becomes one Observation in the parent's current round. A separate subagent step budget (default 30) and invocation budget (default 8 per parent run) bound nested cost; recursion depth is structurally limited to one.

## 7. CLI and configuration

The CLI is one-shot. It accepts a task argument or reads one task from stdin, but does not maintain an interactive session. Configuration includes:

- `--workspace`
- `--provider`
- `--model`
- `--base-url`
- credential source
- `--api-mode` (default `chat_completion`)
- `--max-steps`
- `--context-window`
- `--prompt-dir`

The console entry point is exposed through `pyproject.toml`. Final Agent text goes to stdout; diagnostics go to stderr. Configuration and client failures use a non-zero exit code. Tool errors are handled inside the Agent loop as Observations.

The package targets Python 3.10+ and should avoid 3.11-only syntax or runtime APIs. `pathlib.Path` is used for path handling throughout.

## 8. Testing and acceptance

Tests must cover:

- traversal, absolute out-of-bounds paths, Symlinks, missing files, permissions, deterministic listing, recursive search, line numbers, and the 300-line read cap;
- the 80% Blackboard trigger, latest-five-round retention, immutable system prompt, provenance, atomic failed updates, and system-message injection;
- Chat Completions and Responses normalization, multiple tool calls, malformed responses, and provider errors;
- text completion, tool/Observation cycles, invalid tools, max-step termination, subagent isolation, and recursion prevention;
- one-shot CLI configuration, output streams, and exit codes.

Provider tests use fake clients and mocked adapters. Live provider smoke tests are optional and require external credentials or an available vLLM endpoint.

The implementation quality gate is:

```text
python -m pytest
ruff check .
mypy .
python -m build
```

Each individual code file should target 500 lines or fewer and must never exceed 800 lines. This is a per-file constraint, not an aggregate project limit. The implementation is complete when the package imports, the CLI runs with a fake client, all core acceptance tests pass, and fresh validation evidence is available.

## References

- [Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture](https://arxiv.org/abs/2507.01701)
- [agent-blackboard example implementation](https://github.com/claudioed/agent-blackboard)
