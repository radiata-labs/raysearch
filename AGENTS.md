# AGENTS.md

Project-specific operating rules for all agents working in this repository.

## 1) Project Structure Index (High-Level)

Use this index first to locate files quickly before deep scanning.

```text
.
|- .codex/
|  |- SCOPES.md
|- src/
|  |- answer.py
|  |- fetch.py
|  |- research.py
|  |- search.py
|  |- serpsage/
|     |- app/
|     |- components/
|     |- core/
|     |- models/
|     |- settings/
|     |- steps/
|     |- tokenize/
|- tests/
|- pyproject.toml
|- uv.lock
|- README.md
```

## 2) Python Interpreter and Dependency Rules

- Use `./.venv/Scripts/python.exe` as the only Python interpreter for all Python commands.
- Do not use `pip` directly. Use `uv pip` for package operations.
- When adding or changing dependencies, update `pyproject.toml` (and keep lock state consistent as needed).

## 3) Scope-First Editing Protocol

All edits must be coordinated through `.codex/SCOPES.md`.

### 3.1 Before Editing

- Scan related code and strictly define the exact file scope.
- Append a new task section in `.codex/SCOPES.md` for this work.
- Mark each planned file as `WAITING`.
- Check existing active scope records from other tasks and avoid interference.
- If a target file is occupied, postpone it to the end of the task.
- If it is still occupied when you need to edit it, stop and notify the user.

### 3.2 During Editing

- Stay strictly inside the declared scope for the current task.
- After each file is edited, append a status record for that file as `MODIFYING`.

### 3.3 After Editing (No Tests in This Workflow)

- Do not run tests for this project workflow.
- Review all edited code.
- Run the following commands in order using `./.venv/Scripts/python.exe`:
  1. `./.venv/Scripts/python.exe -m pyright`
  2. `./.venv/Scripts/python.exe -m mypy .`
  3. `./.venv/Scripts/python.exe -m ruff check --fix`
- Fix all reported issues.
- Run the same three commands again in the same order.
- After review passes, append `COMPLETED` records in `.codex/SCOPES.md` for each edited file.
- A file marked `COMPLETED` is no longer occupied.

## 4) Language Requirement

- Keep this system fully in English.
- `.codex/SCOPES.md` must not contain Chinese.
