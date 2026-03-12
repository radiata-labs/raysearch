from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TASK_LOG_HEADER = "## Task Log"
START_MARKER = "<!-- BEGIN AUTO-GENERATED FILE TREE -->"
END_MARKER = "<!-- END AUTO-GENERATED FILE TREE -->"
TASK_HEADER_RE = re.compile(
    r"^### Task: .+ \((?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\)\s*$"
)
STATUS_ROW_RE = re.compile(
    r"^(?P<prefix>\s*-\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+\|\s+`(?P<path>[^`]+)`\s+\|\s+`)(?P<status>WAITING|MODIFYING|COMPLETED)(?P<suffix>`\s+\|\s+.*)$"
)

CONTROL_FILES = {
    ".codex/SCOPES.md",
    ".codex/ARCHIVED.md",
    "AGENTS.md",
}
PYTHON_EXTENSIONS = {".py", ".pyi"}
MYPY_CACHE_DIR = ".mypy_cache_codex"


class FinalizeError(RuntimeError):
    pass


@dataclass(frozen=True)
class StatusEntry:
    line_index: int
    path: str
    status: str


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

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")

    def statuses(self) -> list[StatusEntry]:
        items: list[StatusEntry] = []
        for idx, line in enumerate(self.lines):
            match = STATUS_ROW_RE.match(line)
            if not match:
                continue
            items.append(
                StatusEntry(
                    line_index=idx,
                    path=match.group("path").replace("\\", "/").strip(),
                    status=match.group("status"),
                )
            )
        return items


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _task_log_start(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == TASK_LOG_HEADER:
            return idx
    raise FinalizeError("SCOPES.md is missing '## Task Log' section.")


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
        raise FinalizeError(f"Task not found in SCOPES.md: {task_timestamp}{detail}")
    if len(matches) > 1:
        raise FinalizeError(
            "Multiple tasks matched. Provide --title-contains to disambiguate."
        )
    return matches[0]


def _replace_status(line: str, target_status: str) -> str:
    match = STATUS_ROW_RE.match(line)
    if not match:
        raise FinalizeError("Invalid status row format; cannot update status safely.")
    return f"{match.group('prefix')}{target_status}{match.group('suffix')}"


def _archive_has_timestamp(archive_text: str, task_timestamp: str) -> bool:
    for line in archive_text.split("\n"):
        match = TASK_HEADER_RE.match(line)
        if match and match.group("timestamp") == task_timestamp:
            return True
    return False


def _append_task_to_archive(
    archive_text: str, task_text: str, task_timestamp: str
) -> tuple[str, bool]:
    if _archive_has_timestamp(archive_text, task_timestamp):
        return archive_text, False

    base = archive_text.rstrip("\n")
    section = task_text.strip("\n")
    if not base:
        return section + "\n", True
    return base + "\n\n" + section + "\n", True


def _remove_task_from_scopes(lines: list[str], task: TaskBlock) -> str:
    before = lines[: task.start]
    after = lines[task.end :]
    if before and after and before[-1].strip() == "" and after[0].strip() == "":
        after = after[1:]
    merged = before + after
    return "\n".join(merged).rstrip("\n") + "\n"


def _refresh_agents_tree(repo_root: Path) -> None:
    agents_path = repo_root / "AGENTS.md"
    current = agents_path.read_text(encoding="utf-8")
    tree_text = _build_tree_text(repo_root)
    updated = _replace_tree_block(current, tree_text)
    if updated != current:
        agents_path.write_text(updated, encoding="utf-8", newline="\n")


def _should_skip_tree_path(path: Path, repo_root: Path) -> bool:
    name = path.name
    if name == "__pycache__":
        return True
    if name.startswith(".") and name != ".codex":
        return True
    rel = path.relative_to(repo_root)
    return rel.parts == (".codex", "ARCHIVED.md")


def _sorted_tree_children(path: Path, repo_root: Path) -> list[Path]:
    children = [
        child
        for child in path.iterdir()
        if not _should_skip_tree_path(child, repo_root)
    ]
    children.sort(key=lambda p: (p.is_file(), p.name.lower(), p.name))
    return children


def _build_tree_text(repo_root: Path) -> str:
    lines: list[str] = ["."]

    def _is_in_src_subtree(path: Path) -> bool:
        rel_parts = path.relative_to(repo_root).parts
        return len(rel_parts) > 0 and rel_parts[0] == "src"

    def walk(current: Path, prefix: str) -> None:
        children = _sorted_tree_children(current, repo_root)
        dirs = [child for child in children if child.is_dir()]
        if _is_in_src_subtree(current):
            visible_files: list[Path] = []
        else:
            visible_files = [child for child in children if child.is_file()]
        for child in dirs:
            lines.append(f"{prefix}|- {child.name}/")
            walk(child, prefix + "|  ")
        lines.extend(f"{prefix}|- {child.name}" for child in visible_files)

    walk(repo_root, "")
    return "\n".join(lines)


def _replace_tree_block(content: str, tree_text: str) -> str:
    start = content.find(START_MARKER)
    end = content.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise FinalizeError("AGENTS.md is missing file-tree markers.")
    replacement = f"{START_MARKER}\n```text\n{tree_text}\n```\n{END_MARKER}"
    return content[:start] + replacement + content[end + len(END_MARKER) :]


def _run_cmd(
    repo_root: Path,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    merged_env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(  # noqa: PLW1510, S603
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=merged_env,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


def _run_file_checks(repo_root: Path, rel_path: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    mypy_env = dict(os.environ)
    mypy_env["MYPY_CACHE_DIR"] = MYPY_CACHE_DIR
    commands: list[tuple[list[str], dict[str, str] | None]] = [
        ([sys.executable, "-m", "pyright", rel_path], None),
        ([sys.executable, "-m", "mypy", rel_path], mypy_env),
        ([sys.executable, "-m", "ruff", "check", "--fix", rel_path], None),
        ([sys.executable, "-m", "ruff", "format", rel_path], None),
    ]
    for cmd, env in commands:
        ok, output = _run_cmd(repo_root, cmd, env=env)
        if not ok:
            failures.append(f"$ {' '.join(cmd)}\n{output}".rstrip())
            break
    return len(failures) == 0, failures


def _cleanup_mypy_cache(repo_root: Path) -> None:
    shutil.rmtree(repo_root / MYPY_CACHE_DIR, ignore_errors=True)


def _is_python_file(path: str) -> bool:
    return Path(path).suffix.lower() in PYTHON_EXTENSIONS


def _print_task_list(tasks: list[TaskBlock]) -> None:
    if not tasks:
        print("No task sections found under ## Task Log.")
        return
    print("Tasks in SCOPES.md:")
    for task in tasks:
        statuses = task.statuses()
        waiting = sum(1 for s in statuses if s.status == "WAITING")
        modifying = sum(1 for s in statuses if s.status == "MODIFYING")
        completed = sum(1 for s in statuses if s.status == "COMPLETED")
        print(
            f"- {task.timestamp} | {task.title} | "
            f"WAITING={waiting}, MODIFYING={modifying}, COMPLETED={completed}"
        )


def finalize_task(task_timestamp: str, title_contains: str | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    scopes_path = repo_root / ".codex" / "SCOPES.md"
    archive_path = repo_root / ".codex" / "ARCHIVED.md"

    scopes_text = _normalize_newlines(scopes_path.read_text(encoding="utf-8"))
    scopes_lines = scopes_text.split("\n")
    tasks = _parse_tasks(scopes_lines)
    task = _select_task(tasks, task_timestamp, title_contains)
    status_entries = task.statuses()

    if not status_entries:
        raise FinalizeError(
            f"Task {task.timestamp} has no status rows; cannot finalize safely."
        )

    waiting_count = sum(1 for item in status_entries if item.status == "WAITING")
    if waiting_count > 0:
        raise FinalizeError(
            f"Task contains WAITING entries ({waiting_count}); finish editing first."
        )

    modifying_entries = [item for item in status_entries if item.status == "MODIFYING"]
    if not modifying_entries:
        print("No MODIFYING files left in target task.")
    else:
        print(f"Processing {len(modifying_entries)} MODIFYING file(s)...")

    failures: list[tuple[str, list[str]]] = []

    # Update status file-by-file as each one passes.
    for item in modifying_entries:
        path = item.path
        if path in CONTROL_FILES:
            print(f"SKIP control file: {path}")
            continue

        full_path = repo_root / path
        if not full_path.is_file():
            failures.append((path, [f"Missing file on disk: {path}"]))
            continue

        if not _is_python_file(path):
            print(f"PASS {path} (non-Python file, static checks skipped)")
            task_line_index = task.start + item.line_index
            scopes_lines[task_line_index] = _replace_status(
                scopes_lines[task_line_index], "COMPLETED"
            )
            scopes_path.write_text(
                "\n".join(scopes_lines).rstrip("\n") + "\n",
                encoding="utf-8",
                newline="\n",
            )
            continue

        ok, errs = _run_file_checks(repo_root, path)
        if ok:
            print(f"PASS {path}")
            task_line_index = task.start + item.line_index
            scopes_lines[task_line_index] = _replace_status(
                scopes_lines[task_line_index], "COMPLETED"
            )
            scopes_path.write_text(
                "\n".join(scopes_lines).rstrip("\n") + "\n",
                encoding="utf-8",
                newline="\n",
            )
        else:
            print(f"FAIL {path}")
            failures.append((path, errs))

    # Re-read current task status after incremental updates.
    latest_scopes = _normalize_newlines(scopes_path.read_text(encoding="utf-8"))
    latest_lines = latest_scopes.split("\n")
    latest_task = _select_task(
        _parse_tasks(latest_lines), task_timestamp, title_contains
    )
    latest_statuses = latest_task.statuses()
    has_waiting = any(s.status == "WAITING" for s in latest_statuses)
    has_modifying = any(s.status == "MODIFYING" for s in latest_statuses)

    if failures:
        print("\nScoped check failures:")
        for path, errs in failures:
            print(f"\n[{path}]")
            for err in errs:
                print(err)
        print("\nTask not finalized. Fix failed files and rerun this same command.")
        return 1

    if has_waiting or has_modifying:
        print("Task status is not fully COMPLETED yet; no archival move performed.")
        return 1

    archive_text = _normalize_newlines(archive_path.read_text(encoding="utf-8"))
    updated_archive, appended = _append_task_to_archive(
        archive_text, latest_task.text, latest_task.timestamp
    )
    if appended:
        archive_path.write_text(updated_archive, encoding="utf-8", newline="\n")

    updated_scopes = _remove_task_from_scopes(latest_lines, latest_task)
    scopes_path.write_text(updated_scopes, encoding="utf-8", newline="\n")

    _refresh_agents_tree(repo_root)

    print(
        f"Task finalized: {latest_task.timestamp} | {latest_task.title}. "
        "All files are COMPLETED, task archived, tree refreshed."
    )
    return 0


def main() -> None:
    # _refresh_agents_tree(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(
        description=(
            "Run per-file scoped checks for one task, update status rows "
            "MODIFYING->COMPLETED file-by-file, then archive when fully completed."
        )
    )
    parser.add_argument(
        "task_timestamp",
        nargs="?",
        help="Task timestamp in header format: YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--title-contains",
        help="Optional title fragment to disambiguate tasks with the same timestamp.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current tasks under ## Task Log with status counters.",
    )
    parser.add_argument(
        "--refresh-tree-only",
        action="store_true",
        help="Refresh AGENTS file tree block and exit.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    try:
        scopes_path = repo_root / ".codex" / "SCOPES.md"
        scopes_text = _normalize_newlines(scopes_path.read_text(encoding="utf-8"))
        tasks = _parse_tasks(scopes_text.split("\n"))

        if args.refresh_tree_only:
            _refresh_agents_tree(repo_root)
            print("AGENTS tree refreshed.")
            return

        if args.list:
            _print_task_list(tasks)
            return
        if not args.task_timestamp:
            raise FinalizeError("task_timestamp is required unless --list is used.")

        raise SystemExit(finalize_task(args.task_timestamp, args.title_contains))
    except FinalizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        _cleanup_mypy_cache(repo_root)


if __name__ == "__main__":
    main()
