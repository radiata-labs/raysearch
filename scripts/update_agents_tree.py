from __future__ import annotations

from pathlib import Path

START_MARKER = "<!-- BEGIN AUTO-GENERATED FILE TREE -->"
END_MARKER = "<!-- END AUTO-GENERATED FILE TREE -->"


def _should_skip(path: Path, repo_root: Path) -> bool:
    name = path.name
    if name == "__pycache__":
        return True
    if name.startswith(".") and name != ".codex":
        return True
    rel = path.relative_to(repo_root)
    return rel.parts == (".codex", "ARCHIVED.md")


def _sorted_children(path: Path, repo_root: Path) -> list[Path]:
    children = [child for child in path.iterdir() if not _should_skip(child, repo_root)]
    children.sort(key=lambda p: (p.is_file(), p.name.lower(), p.name))
    return children


def build_tree_text(repo_root: Path) -> str:
    lines: list[str] = ["."]

    def _is_in_src_subtree(path: Path) -> bool:
        rel_parts = path.relative_to(repo_root).parts
        return len(rel_parts) > 0 and rel_parts[0] == "src"

    def walk(current: Path, prefix: str) -> None:
        children = _sorted_children(current, repo_root)
        dirs = [child for child in children if child.is_dir()]

        # Keep `src` concise: show directory structure only.
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


def replace_tree_block(content: str, tree_text: str) -> str:
    start = content.find(START_MARKER)
    end = content.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise ValueError("AGENTS.md is missing file-tree markers.")

    replacement = f"{START_MARKER}\n```text\n{tree_text}\n```\n{END_MARKER}"
    return content[:start] + replacement + content[end + len(END_MARKER) :]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    agents_path = repo_root / "AGENTS.md"

    current = agents_path.read_text(encoding="utf-8")
    tree_text = build_tree_text(repo_root)
    updated = replace_tree_block(current, tree_text)

    if updated != current:
        agents_path.write_text(updated, encoding="utf-8", newline="\n")
        print("Updated AGENTS.md file-tree block.")
    else:
        print("AGENTS.md file-tree block is already up to date.")


if __name__ == "__main__":
    main()
