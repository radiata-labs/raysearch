from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TASK_HEADER_RE = re.compile(
    r"^### Task: .+ \((?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\)\s*$"
)
STATUS_RE = re.compile(r"\|\s*`(?P<status>WAITING|MODIFYING|COMPLETED)`\s*\|")


class FinalizeError(RuntimeError):
    pass


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _find_task_bounds(lines: list[str], task_timestamp: str) -> tuple[int, int]:
    starts: list[int] = []
    target_start: int | None = None
    for idx, line in enumerate(lines):
        match = TASK_HEADER_RE.match(line)
        if not match:
            continue
        starts.append(idx)
        if match.group("timestamp") == task_timestamp:
            if target_start is not None:
                raise FinalizeError(
                    f"Multiple tasks found for timestamp: {task_timestamp}"
                )
            target_start = idx

    if target_start is None:
        raise FinalizeError(f"Task not found in SCOPES.md: {task_timestamp}")

    next_starts = [s for s in starts if s > target_start]
    target_end = min(next_starts) if next_starts else len(lines)
    return target_start, target_end


def _validate_task_statuses(task_lines: list[str], task_timestamp: str) -> None:
    statuses: list[str] = []
    for line in task_lines:
        match = STATUS_RE.search(line)
        if match:
            statuses.append(match.group("status"))

    if not statuses:
        raise FinalizeError(
            f"Task {task_timestamp} has no status rows; cannot finalize safely."
        )

    waiting = sum(1 for s in statuses if s == "WAITING")
    modifying = sum(1 for s in statuses if s == "MODIFYING")
    if waiting > 0 or modifying > 0:
        raise FinalizeError(
            f"Task is not fully COMPLETED: WAITING={waiting}, MODIFYING={modifying}."
        )


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


def _remove_task_from_scopes(lines: list[str], start: int, end: int) -> str:
    before = lines[:start]
    after = lines[end:]
    if before and after and before[-1].strip() == "" and after[0].strip() == "":
        after = after[1:]
    merged = before + after
    return "\n".join(merged).rstrip("\n") + "\n"


def _refresh_agents_tree(repo_root: Path) -> None:
    tree_script = repo_root / "scripts" / "update_agents_tree.py"
    if not tree_script.is_file():
        raise FinalizeError(f"Script not found: {tree_script}")
    subprocess.run([sys.executable, str(tree_script)], cwd=repo_root, check=True)  # noqa: S603


def finalize_task(task_timestamp: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scopes_path = repo_root / ".codex" / "SCOPES.md"
    archive_path = repo_root / ".codex" / "ARCHIVED.md"

    scopes_text = _normalize_newlines(scopes_path.read_text(encoding="utf-8"))
    archive_text = _normalize_newlines(archive_path.read_text(encoding="utf-8"))

    scopes_lines = scopes_text.split("\n")
    start, end = _find_task_bounds(scopes_lines, task_timestamp)
    task_lines = scopes_lines[start:end]
    task_text = "\n".join(task_lines).strip("\n")

    _validate_task_statuses(task_lines, task_timestamp)

    updated_archive, appended = _append_task_to_archive(
        archive_text, task_text, task_timestamp
    )
    if appended:
        archive_path.write_text(updated_archive, encoding="utf-8", newline="\n")

    updated_scopes = _remove_task_from_scopes(scopes_lines, start, end)
    scopes_path.write_text(updated_scopes, encoding="utf-8", newline="\n")

    _refresh_agents_tree(repo_root)

    action = "appended+moved" if appended else "moved (already archived)"
    print(f"Task finalized: {task_timestamp} ({action}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize one SCOPES task by timestamp, validate statuses, archive it, "
            "and refresh AGENTS tree."
        )
    )
    parser.add_argument(
        "task_timestamp",
        help="Task timestamp in header format: YYYY-MM-DD HH:MM:SS",
    )
    args = parser.parse_args()

    try:
        finalize_task(args.task_timestamp)
    except FinalizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
