from difflib import unified_diff
from pathlib import Path
from fnmatch import fnmatch


def _read_gitignore_patterns(gitignore_path):
    patterns = []

    if not gitignore_path.exists():
        return patterns

    for line in gitignore_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        negated = line.startswith("!")
        if negated:
            line = line[1:]

        if line:
            patterns.append((line, negated))

    return patterns


def _matches_gitignore_pattern(relative_path, pattern):
    path = relative_path.as_posix()
    is_dir_pattern = pattern.endswith("/")
    anchored = pattern.startswith("/")

    pattern = pattern.strip("/")
    if not pattern:
        return False

    if is_dir_pattern:
        if anchored:
            return path == pattern or path.startswith(pattern + "/")
        return any(part == pattern for part in relative_path.parts)

    if "/" in pattern or anchored:
        return fnmatch(path, pattern)

    return any(fnmatch(part, pattern) for part in relative_path.parts)


def _is_gitignored(relative_path, patterns):
    ignored = False

    for pattern, negated in patterns:
        if _matches_gitignore_pattern(relative_path, pattern):
            ignored = not negated

    return ignored


def get_current_source():
    root = Path(".")
    patterns = _read_gitignore_patterns(root / ".gitignore")
    source = {}

    for path in root.rglob("*.py"):
        relative_path = path.relative_to(root)
        if _is_gitignored(relative_path, patterns):
            continue

        source[relative_path.as_posix()] = path.read_text()

    return source


def _source_stats(contents):
    return {
        "lines": len(contents.splitlines()),
        "chars": len(contents),
    }


def source_dict_diff(before, after):
    before_paths = set(before)
    after_paths = set(after)

    added = {
        path: _source_stats(after[path])
        for path in sorted(after_paths - before_paths)
    }
    removed = {
        path: _source_stats(before[path])
        for path in sorted(before_paths - after_paths)
    }
    changed = {}

    for path in sorted(before_paths & after_paths):
        before_contents = before[path]
        after_contents = after[path]
        if before_contents == after_contents:
            continue

        diff_lines = unified_diff(
            before_contents.splitlines(),
            after_contents.splitlines(),
            fromfile=f"before/{path}",
            tofile=f"after/{path}",
            lineterm="",
        )
        changed[path] = {
            "before": _source_stats(before_contents),
            "after": _source_stats(after_contents),
            "diff": "\n".join(diff_lines),
        }

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
    }



