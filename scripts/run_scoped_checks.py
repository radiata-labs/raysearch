from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TASK_LOG_HEADER = "## Task Log"
TASK_HEADER_RE = re.compile(
    r"^### Task: .+ \((?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\)\s*$"
)
STATUS_ROW_RE = re.compile(
    r"^\s*-\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+\|\s+`(?P<path>[^`]+)`\s+\|\s+`(?P<status>WAITING|MODIFYING|COMPLETED)`\s+\|"
)

CONTROL_FILES = {
    ".codex/SCOPES.md",
    ".codex/ARCHIVED.md",
    "AGENTS.md",
}
PYTHON_EXTENSIONS = {".py", ".pyi"}


class ScopedCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskBlock:
    header_line: str
    timestamp: str
    start: int
    end: int
    lines: list[str]

    @property
    def title(self) -> str:
        return self.header_line.removeprefix("### Task: ").rsplit(" (", 1)[0]


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _task_log_start(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == TASK_LOG_HEADER:
            return idx
    raise ScopedCheckError("SCOPES.md is missing '## Task Log' section.")


def _parse_tasks(lines: list[str]) -> list[TaskBlock]:
    start_idx = _task_log_start(lines) + 1
    starts: list[tuple[int, str, str]] = []

    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        match = TASK_HEADER_RE.match(line)
        if not match:
            continue
        starts.append((idx, line, match.group("timestamp")))

    tasks: list[TaskBlock] = []
    for i, (task_start, header_line, timestamp) in enumerate(starts):
        task_end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        tasks.append(
            TaskBlock(
                header_line=header_line,
                timestamp=timestamp,
                start=task_start,
                end=task_end,
                lines=lines[task_start:task_end],
            )
        )
    return tasks


def _select_task(
    tasks: list[TaskBlock], task_timestamp: str, title_contains: str | None
) -> TaskBlock:
    matches = [task for task in tasks if task.timestamp == task_timestamp]
    if title_contains:
        lowered = title_contains.lower()
        matches = [task for task in matches if lowered in task.title.lower()]

    if not matches:
        detail = f" and title containing '{title_contains}'" if title_contains else ""
        raise ScopedCheckError(f"Task not found in SCOPES.md: {task_timestamp}{detail}")
    if len(matches) > 1:
        raise ScopedCheckError(
            "Multiple tasks matched. Provide --title-contains to disambiguate."
        )
    return matches[0]


def _collect_modified_files(task: TaskBlock) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()

    for line in task.lines:
        match = STATUS_ROW_RE.match(line)
        if not match:
            continue
        path = match.group("path").strip()
        status = match.group("status")

        if status == "WAITING":
            continue
        normalized = path.replace("\\", "/")
        if normalized in CONTROL_FILES:
            continue
        if normalized not in seen:
            seen.add(normalized)
            files.append(normalized)

    if not files:
        raise ScopedCheckError(
            "No modified files found for this task (non-control files with status "
            "MODIFYING/COMPLETED)."
        )
    return files


def _assert_files_exist(repo_root: Path, files: list[str]) -> None:
    missing = [path for path in files if not (repo_root / path).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise ScopedCheckError(f"Scoped files not found on disk: {joined}")


def _python_files_only(files: list[str]) -> list[str]:
    return [path for path in files if Path(path).suffix.lower() in PYTHON_EXTENSIONS]


def _run(repo_root: Path, args: list[str]) -> None:
    subprocess.run(args, cwd=repo_root, check=True)  # noqa: S603


def _run_checks(repo_root: Path, files: list[str], passes: int) -> None:
    pyright_cmd = [sys.executable, "-m", "pyright", *files]
    mypy_cmd = [sys.executable, "-m", "mypy", *files]
    ruff_check_cmd = [sys.executable, "-m", "ruff", "check", "--fix", *files]
    ruff_format_cmd = [sys.executable, "-m", "ruff", "format", *files]

    for index in range(1, passes + 1):
        print(f"Scoped check pass {index}/{passes}")
        _run(repo_root, pyright_cmd)
        _run(repo_root, mypy_cmd)
        _run(repo_root, ruff_check_cmd)
        _run(repo_root, ruff_format_cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run pyright/mypy/ruff only on files modified by one SCOPES task "
            "(derived from status rows)."
        )
    )
    parser.add_argument(
        "task_timestamp",
        help="Task timestamp in header format: YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--title-contains",
        help="Optional title fragment to disambiguate tasks with the same timestamp.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Number of sequential passes (default: 2).",
    )
    args = parser.parse_args()

    if args.passes < 1:
        raise SystemExit("ERROR: --passes must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    scopes_path = repo_root / ".codex" / "SCOPES.md"
    scopes_text = _normalize_newlines(scopes_path.read_text(encoding="utf-8"))
    tasks = _parse_tasks(scopes_text.split("\n"))

    try:
        task = _select_task(tasks, args.task_timestamp, args.title_contains)
        files = _collect_modified_files(task)
        _assert_files_exist(repo_root, files)
        python_files = _python_files_only(files)
        print(
            "Scoped files:",
            ", ".join(files),
        )
        if not python_files:
            print("No Python files in scoped set; static checks skipped.")
            return
        print("Scoped Python files:", ", ".join(python_files))
        _run_checks(repo_root, python_files, passes=args.passes)
    except ScopedCheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
