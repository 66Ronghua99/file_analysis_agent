# Local File Analysis Agent Implementation Plan

## Goal Description

Implement a Python 3.10+ package and one-shot CLI for a local, read-only file analysis Agent. The implementation must enforce an application-level `workspace_dir` boundary, expose safe directory and text-reading tools through external Function Calling schemas, support OpenRouter and external vLLM through an OpenAI-compatible client plus a LiteLLM adapter, and execute a provider-independent synchronous ReAct loop.

The Agent must maintain a private structured Blackboard. At 80% of the configured context window it retains the latest five complete ReAct rounds, compresses older rounds into the Blackboard through a dedicated updater, and injects the unchanged root system prompt plus the Blackboard packet into subsequent calls. `call_subagent` creates an independent Agent loop with its own system prompt, messages, Blackboard, and bounded execution budget.

The first version is intentionally read-only and one-shot. It does not write files, persist memory across invocations, manage vLLM, provide an interactive CLI, or introduce an operating-system sandbox.

## Acceptance Criteria

- AC-1: The repository contains an installable Python 3.10+ package with a one-shot CLI and a typed configuration boundary.
  - Positive Tests (expected to PASS):
    - The package imports successfully under Python 3.10+ without requiring provider-specific optional dependencies.
    - The CLI exposes `--workspace`, `--provider`, `--model`, `--base-url`, `--api-mode`, `--max-steps`, `--context-window`, and `--prompt-dir`.
    - A fake-client CLI smoke test accepts one task argument and exits successfully with the final Agent text on stdout.
  - Negative Tests (expected to FAIL/reject):
    - A relative, missing, or non-directory workspace configuration is rejected before Agent execution.
    - An unknown provider, missing required model, invalid API mode, or invalid numeric limit returns a non-zero CLI exit code with a diagnostic on stderr.
  - AC-1.1: The implementation targets Python 3.10+ and keeps provider dependencies optional.
    - Positive: A core-package test runs with fake clients and without OpenRouter, vLLM, or LiteLLM credentials.
    - Negative: Importing the core package must not fail solely because an optional provider package is absent.

- AC-2: Every filesystem operation is constrained to the resolved workspace root.
  - Positive Tests (expected to PASS):
    - Relative and absolute paths that resolve inside the workspace are accepted.
    - A Symlink whose target is inside the workspace can be resolved and read according to the tool contract.
    - The workspace root is resolved once with `strict=True` and is confirmed to be a directory.
  - Negative Tests (expected to FAIL/reject):
    - `../` traversal, an external absolute path, and a Symlink targeting outside the workspace all return `[Error] Permission Denied: out of bounds`.
    - No file tool can bypass the shared resolver by constructing an independent path from raw user input.
    - A concurrent link-race threat is documented as outside the application-level boundary rather than being silently described as OS isolation.

- AC-3: The read-only filesystem tools provide deterministic, bounded, LLM-readable results.
  - Positive Tests (expected to PASS):
    - `list_dir` returns direct children in stable sorted order and reports relative paths or clear entry markers consistently.
    - `search_text` recursively finds UTF-8 matches and includes workspace-relative paths, 1-based line numbers, and matching lines.
    - `read_file` supports 1-based pagination and formats each returned line as `<line number> | <content>`.
    - A normal `read_file` request returns no more than 300 lines.
  - Negative Tests (expected to FAIL/reject):
    - A request for more than 300 lines is clamped and includes an explicit truncation notice rather than returning an unbounded payload.
    - Missing paths, permission failures, wrong file types, invalid line ranges, and UTF-8 decode failures return stable error strings without uncaught filesystem exceptions.
    - `search_text` stops at the configurable default maximum of 200 matches and reports truncation instead of flooding the context.
    - Recursive search does not follow an external Directory Symlink.

- AC-4: Tool schemas and Python tool implementations are separately maintained and safely dispatched.
  - Positive Tests (expected to PASS):
    - `file_tools.json` is loaded as a package resource or explicit prompt/resource override and exposes schemas for `list_dir`, `search_text`, `read_file`, and `call_subagent`.
    - `ToolRegistry` validates JSON arguments before invoking the typed Python callable and preserves tool-call IDs in Observations.
    - A prompt directory override changes resource content without changing Agent loop code.
  - Negative Tests (expected to FAIL/reject):
    - An unknown tool name and malformed arguments produce an Observation error and do not execute an arbitrary callable.
    - The Agent loop does not contain hard-coded system prompt text or Function Calling schema definitions.
    - A subagent registry rejects `call_subagent` as unavailable.

- AC-5: OpenRouter, external vLLM, and LiteLLM are represented by provider-independent normalized client responses.
  - Positive Tests (expected to PASS):
    - An OpenAI-compatible fake response for OpenRouter and an external vLLM base URL normalizes to the same internal `LLMResponse` shape.
    - A LiteLLM fake response normalizes to the same shape.
    - Chat Completions is the default mode, and supported Responses-format payloads are normalized without changing the Agent loop.
    - Multiple tool calls in one response preserve call IDs, names, and JSON arguments.
  - Negative Tests (expected to FAIL/reject):
    - A provider or model that does not support the selected API mode returns an explicit client error and does not silently switch modes.
    - Malformed provider responses, missing tool-call fields, and transport/configuration failures produce diagnosable errors.
    - The Agent loop contains no provider-specific branches or raw SDK response parsing.

- AC-6: The ReAct loop completes ordinary tasks, feeds tool results back as Observations, and terminates safely.
  - Positive Tests (expected to PASS):
    - A fake client returning final assistant text completes the task in one decision round.
    - A fake client returning a tool call receives the tool result in the next request and can then return final text.
    - Multiple tool calls are validated and executed sequentially, with each Observation appended in order.
    - The default main loop limit is 100 LLM decision rounds and is configurable.
  - Negative Tests (expected to FAIL/reject):
    - Unknown tools, invalid arguments, and filesystem errors are returned to the model as Observations instead of crashing the process.
    - A fake client that never returns final text is stopped at `max_steps` with a clear termination error.
    - Provider failures and Blackboard update failures terminate explicitly rather than entering an uncontrolled retry loop.

- AC-7: Blackboard context compression preserves the immutable system prompt, the latest five ReAct rounds, and a validated structured history.
  - Positive Tests (expected to PASS):
    - A fake `TokenCounter` crossing 80% triggers compression before the next model request.
    - The active unfinished round and the latest five complete ReAct rounds remain available after compression.
    - Older rounds are represented in a structured Blackboard containing task, facts/findings, decisions, open questions, progress, next action, and provenance where available.
    - The exact root `system.md` content is unchanged and a serialized Blackboard packet is injected as a separate system message.
    - A successful update is committed atomically and later compression merges with the existing Blackboard.
    - A deterministic approximate token counter is available by default and a provider-specific counter can be injected.
  - Negative Tests (expected to FAIL/reject):
    - An invalid Blackboard updater payload does not replace the existing Blackboard or delete raw history.
    - If the Blackboard plus the latest five rounds cannot fit inside the configured context window, the Agent returns a context-budget error instead of silently dropping the active round or state.
    - Blackboard serialization is bounded by its approximately 20% default budget and does not grow without limit.

- AC-8: `call_subagent` executes an independent bounded Agent.
  - Positive Tests (expected to PASS):
    - A subagent uses `subagent_system.md`, a fresh message history, a fresh Blackboard, and the same client configuration by default.
    - The subagent can use filesystem tools and return a final result to the parent.
    - The parent records the subagent final result as one Observation in its current ReAct round.
    - The default subagent budget is 30 decision rounds and 8 invocations per parent run, with both values configurable.
  - Negative Tests (expected to FAIL/reject):
    - The subagent cannot call `call_subagent` and cannot recursively create another subagent.
    - Parent messages and Blackboard mutations are not visible as live shared state inside the subagent.
    - A subagent that exceeds its budget returns a bounded error Observation rather than running indefinitely.

- AC-9: The one-shot CLI has predictable input, output, and failure behavior.
  - Positive Tests (expected to PASS):
    - A task passed as an argument runs exactly once and writes final Agent text to stdout.
    - A task read from stdin runs exactly once when no task argument is supplied.
    - Diagnostics are written to stderr and do not contaminate the final stdout payload.
    - A console entry point is declared in `pyproject.toml` and package resources are available after installation.
  - Negative Tests (expected to FAIL/reject):
    - Supplying both an argument task and stdin-only mode, or supplying no task, is rejected clearly.
    - Configuration and client failures use non-zero exit codes.
    - The CLI does not enter an interactive session or persist Memory between invocations.

- AC-10: The implementation is maintainable, testable, and respects the project quality gate.
  - Positive Tests (expected to PASS):
    - `python -m pytest` passes with filesystem, client, Blackboard, loop, subagent, and CLI coverage.
    - `ruff check .`, `mypy .`, and `python -m build` pass for the supported package.
    - Production and test code use type hints and `pathlib.Path` for path handling.
    - Every individual code file is at or below 800 lines, with a target of 500 lines or fewer.
  - Negative Tests (expected to FAIL/reject):
    - A file-size check fails when any code file exceeds the 800-line hard limit.
    - A lint/type/build failure is not replaced by an old or external validation result.
    - The implementation must not add LangChain, file-writing tools, persistent Memory, an interactive CLI, or OS-level sandbox claims within this scope.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

Implement all approved package modules, external prompt and schema resources, OpenRouter/vLLM and LiteLLM adapters, the one-shot CLI, deterministic fake-client tests, mocked provider adapter tests, Blackboard compaction tests, filesystem security tests, and the complete local quality gate. Provide clear package extras and documentation sufficient to run the CLI against an external vLLM endpoint or supported provider.

### Lower Bound (Minimum Acceptable Scope)

Implement the installable package, safe read-only tools, external schemas/prompts, normalized client protocol with the three required backend paths, Blackboard compression, independent subagent loop, one-shot CLI, and deterministic tests for every acceptance criterion. Live network smoke tests, provider-specific tokenizer integration, persistent storage, file writes, and interactive UX are not required for the minimum scope.

### Allowed Choices

- Can use: Python standard library, `pathlib`, `dataclasses`, `typing.Protocol`, `importlib.resources`, a lightweight schema validator such as Pydantic or `jsonschema`, `openai` for OpenAI-compatible endpoints, LiteLLM as an optional extra, pytest, Ruff, mypy, and build tooling.
- Can use: synchronous provider adapters and fake/mock clients for deterministic tests.
- Can use: a provider-specific tokenizer later, behind the `TokenCounter` protocol.
- Cannot use: LangChain or another heavy Agent orchestration framework.
- Cannot use: direct raw paths outside the shared Sandbox resolver.
- Cannot add: file writing/editing/deletion, interactive CLI sessions, persistent Memory, vLLM process management, or OS/container sandboxing.
- Must use: `pathlib.Path` for path handling, external prompt/schema resources, explicit error returns for expected filesystem failures, and dependency injection at provider/tool/memory boundaries.

## Feasibility Hints and Suggestions

### Conceptual Approach

Build bottom-up from stable contracts:

1. Define typed internal messages, tool calls, completion requests/responses, configuration, and error models.
2. Implement `SandboxResolver` once and make every filesystem operation depend on it.
3. Implement filesystem tools and bind them to external JSON schemas through `ToolRegistry`.
4. Implement the normalized client protocol and adapter-specific parsing behind it; test adapters with fixtures before connecting the loop.
5. Implement the Blackboard as a per-Agent mutable object, plus an updater that validates structured JSON before atomically replacing state.
6. Implement the loop with a fake client first, then wire in the provider factory and subagent factory.
7. Add the one-shot CLI last, using the same dependency-injected factory as the package API.

The context assembly order should remain stable: immutable root system prompt, serialized Blackboard system packet, then the latest raw conversation data. Compression must complete successfully before older rounds are deleted.

Use `importlib.resources` for default package resources and allow a prompt directory override at the configuration boundary. Keep provider SDK imports local to their adapter modules so core imports remain usable without optional extras.

Permission failure tests may need monkeypatching or a controlled fake filesystem error rather than relying only on OS permissions, because test environments can run with elevated privileges. Symlink tests should use `tmp_path` and explicitly create both in-bound and out-of-bound targets.

### Relevant References

- `docs/superpowers/specs/2026-07-14-local-file-analysis-agent-design.md` - approved architecture, contracts, scope, and defaults.
- `file_analysis_agent/sandbox/resolver.py` - planned single path-security boundary.
- `file_analysis_agent/tools/filesystem.py` - planned read-only tool implementations.
- `file_analysis_agent/clients/protocol.py` - planned normalized client contract.
- `file_analysis_agent/memory/blackboard.py` - planned context compression state.
- `file_analysis_agent/agent/loop.py` - planned ReAct orchestration boundary.
- `file_analysis_agent/resources/prompts/` - planned externally editable prompt resources.
- `file_analysis_agent/resources/tools/file_tools.json` - planned Function Calling schemas.

## Dependencies and Sequence

### Milestones

1. **Package and contract foundation**
   - Add `pyproject.toml`, package metadata, optional provider extras, development tools, and source package directories.
   - Add typed models, configuration validation, package-resource loading, default prompts, and tool schema resources.
   - Establish the public API exports and a fake client fixture.

2. **Sandbox and read-only tools**
   - Implement the strict workspace root and unified resolver.
   - Implement `list_dir`, `search_text`, and `read_file` with bounded output and stable error strings.
   - Implement schema-backed tool registration and dispatch.
   - Add traversal, Symlink, pagination, decode, and error-path tests.

3. **Provider clients**
   - Define normalized `LLMClient`, request, response, and tool-call models.
   - Implement OpenAI-compatible OpenRouter/vLLM adapter and LiteLLM adapter.
   - Add Chat Completions default behavior, Responses normalization, optional dependency guards, and mocked response fixtures.

4. **Blackboard memory**
   - Implement Blackboard state and provenance-bearing entries.
   - Implement the deterministic approximate token counter and injection protocol.
   - Implement validated Blackboard updates at the 80% threshold, latest-five-round retention, budget enforcement, and atomic failure behavior.
   - Add focused compaction tests before integrating the loop.

5. **Agent loop and subagent**
   - Implement the synchronous ReAct loop with max-step termination and sequential tool calls.
   - Implement the independent subagent factory and restricted registry.
   - Integrate Blackboard context assembly and `call_subagent` Observations.
   - Add fake-client loop, budget, error, and isolation tests.

6. **One-shot CLI and integration**
   - Implement CLI argument/stdin handling, provider factory, prompt override, stdout/stderr behavior, and exit codes.
   - Add package installation and fake-client CLI smoke tests.
   - Update README with install, configuration, and one-task usage examples.

7. **Quality hardening and handoff**
   - Run the full pytest, Ruff, mypy, and build gate.
   - Review all code files against the 500-line target and 800-line hard limit.
   - Review acceptance criteria against fresh test output and record any environment-blocked live smoke tests explicitly.

Dependencies flow from contract foundation to sandbox/tools and clients, then to Blackboard, then to the Agent loop/subagent, and finally to the CLI integration. Tests should be added with each milestone rather than deferred to the end.

## Task Breakdown

Each task has exactly one routing tag. `coding` identifies implementation work; `analyze` identifies repository-level analysis or verification work.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|--------------------------|------------|
| task-01 | Confirm the Python 3.10+ packaging baseline, optional dependency strategy, console entry point, and test/lint/type/build commands against the empty repository. | AC-1, AC-10 | analyze | - |
| task-02 | Add `pyproject.toml`, package directories, public exports, typed configuration, package-resource loading, default prompt files, and `file_tools.json`. | AC-1, AC-4 | coding | task-01 |
| task-03 | Add internal typed models for messages, tool calls, completion requests/responses, execution results, and domain errors. | AC-1, AC-5, AC-6 | coding | task-02 |
| task-04 | Implement the strict `workspace_dir` resolver and sandbox error mapping using `pathlib.Path`. | AC-2 | coding | task-03 |
| task-05 | Implement `list_dir`, `search_text`, and `read_file` with bounds, line numbering, pagination, Symlink handling, and deterministic output. | AC-3 | coding | task-04 |
| task-06 | Implement schema loading, argument validation, callable binding, tool-call IDs, and restricted registry construction. | AC-4, AC-8 | coding | task-02, task-03, task-05 |
| task-07 | Implement the normalized synchronous client protocol and OpenAI-compatible adapter for OpenRouter and external vLLM. | AC-5 | coding | task-03 |
| task-08 | Implement the LiteLLM adapter, optional dependency errors, API-mode validation, and provider response fixtures. | AC-5 | coding | task-03 |
| task-09 | Implement `BlackboardState`, provenance entries, approximate token counting, system-packet serialization, and injectable updater contracts. | AC-7 | coding | task-03, task-02 |
| task-10 | Implement threshold-triggered Blackboard compaction, latest-five-round retention, budget enforcement, structured update validation, and atomic failure behavior. | AC-7 | coding | task-09 |
| task-11 | Implement the synchronous ReAct loop, sequential tool execution, Observation handling, max-step termination, and explicit provider/internal failure paths. | AC-6 | coding | task-06, task-07, task-08, task-10 |
| task-12 | Implement independent subagent construction with its own prompt, loop, Blackboard, restricted registry, step budget, invocation budget, and parent Observation result. | AC-8 | coding | task-06, task-10, task-11 |
| task-13 | Implement the one-shot CLI, provider factory, argument/stdin task input, prompt override, output streams, exit codes, and console entry point integration. | AC-1, AC-9 | coding | task-02, task-07, task-08, task-11, task-12 |
| task-14 | Add unit and integration tests for sandbox/tools, schema dispatch, normalized clients, Blackboard compression, ReAct loop, subagent isolation, and CLI behavior. | AC-2 through AC-9 | coding | task-04 through task-13 |
| task-15 | Run fresh quality-gate verification, inspect per-file line counts, review acceptance evidence, and document any unavailable live-provider smoke tests. | AC-10 | analyze | task-14 |

## Claude-Codex Deliberation

### Agreements

- The approved design is a small Python package plus a one-shot CLI.
- File operations are read-only and must pass through one resolved workspace boundary.
- OpenRouter and external vLLM use an OpenAI-compatible adapter; LiteLLM has a separate adapter.
- The Agent loop is synchronous and provider-independent.
- Main Agent and subagent have separate system prompts, message histories, and Blackboards.
- Blackboard compression triggers at 80%, preserves the latest five ReAct rounds, and keeps the root system prompt unchanged.
- Python 3.10+ compatibility and per-file 500-line target/800-line hard limit are required.

### Resolved Disagreements

- **Memory representation**: A plain conversation summary was replaced by an Agent-owned structured Blackboard after design review. This preserves task state, findings, decisions, open questions, progress, next action, and provenance while allowing incremental updates.
- **Sandbox meaning**: The design explicitly defines application-level path isolation and does not claim OS-level isolation or protection against concurrent filesystem link races.
- **Provider architecture**: Provider-specific branches inside the loop were rejected in favor of a normalized client protocol and adapters.

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- None. The plan follows the approved design spec and records configurable numeric defaults separately from hard acceptance constraints.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive domain names such as `SandboxResolver`, `BlackboardState`, `CompletionRequest`, and `ToolRegistry` in code.
- Keep modules cohesive, use complete type hints compatible with Python 3.10+, and preserve the shared workspace resolver as the only filesystem path boundary.
