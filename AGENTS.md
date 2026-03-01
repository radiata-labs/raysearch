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
- `.codex/SCOPES.md` format is strict: keep canonical headings/order and status record template exactly as defined in that file.
- If scope format drift is detected, fix the format first, then continue with task edits.
- Use second-precision timestamps in scope records: `YYYY-MM-DD HH:MM:SS`.
- Keep one active status record per file per task.
- Status changes are updates, not appends: `WAITING -> MODIFYING -> COMPLETED`.
- Keep status row format exact: `<YYYY-MM-DD HH:MM:SS> | \`<path>\` | \`<status>\` | <note>`.

### 3.1 Before Editing

- Scan related code and strictly define the exact file scope.
- Append a new task section in `.codex/SCOPES.md` for this work.
- Mark each planned file as `WAITING`.
- Check existing active scope records from other tasks and avoid interference.
- If a target file is occupied, postpone it to the end of the task.
- If it is still occupied when you need to edit it, stop and notify the user.
- You may only define scope entries for files you are going to edit in this task.

### 3.2 During Editing

- Stay strictly inside the declared scope for the current task.
- After each file is edited, update its status from `WAITING` to `MODIFYING`.
- You may only update status records that you created in the current task section.

### 3.3 After Editing (No Tests in This Workflow)

- Do not run tests for this project workflow.
- Review all edited code.
- Run the following commands in order using `./.venv/Scripts/python.exe`:
  1. `./.venv/Scripts/python.exe -m pyright`
  2. `./.venv/Scripts/python.exe -m mypy .`
  3. `./.venv/Scripts/python.exe -m ruff check --fix`
  4. `./.venv/Scripts/python.exe -m ruff format`
- Fix all reported issues.
- Run the same four commands again in the same order.
- After review passes, update each file status to `COMPLETED`.
- A file marked `COMPLETED` is no longer occupied.
- When all files in the current task are `COMPLETED` and final checks are done, move the entire task section to `.codex/ARCHIVED.md`.
- Move order is strict: append the exact task section to `.codex/ARCHIVED.md` first, then remove it from `.codex/SCOPES.md`.
- `.codex/ARCHIVED.md` is append-only. Never edit or delete previously archived entries.
- STRICT RULE: ONLY modify or delete sections and records that YOU created. Never modify or delete content created by others.

## 4) Language Requirement

- Keep this system fully in English.
- `.codex/SCOPES.md` must not contain Chinese.
