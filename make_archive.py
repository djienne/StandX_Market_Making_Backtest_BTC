from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".cargo",
    ".github",
    ".gemini",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    ".ipynb_checkpoints",
    ".idea",
    ".vscode",
}

DEFAULT_EXCLUDE_DIR_PATTERNS = [
    "*.egg-info",
]

DEFAULT_EXCLUDE_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.obj",
    "*.o",
    "*.a",
    "*.lib",
    "*.rlib",
    "*.rmeta",
    "*.d",
    "*.tmp",
    "*.log",
    "*.cache",
    "*.npz",
    "*.npz.meta.json",
    "*.zip",
    "*.swp",
    "*.swo",
    "Thumbs.db",
    "desktop.ini",
    ".DS_Store",
]


def _match_name(name: str, patterns: list[str]) -> bool:
    return any(PurePosixPath(name).match(pattern) for pattern in patterns)


def _match_path(path: Path, patterns: list[str]) -> bool:
    rel = PurePosixPath(path.as_posix())
    return any(rel.match(pattern) for pattern in patterns)


def _should_skip_dir(
    rel_path: Path,
    exclude_dirs: set[str],
    exclude_dir_patterns: list[str],
    exclude_paths: list[str],
) -> bool:
    name = rel_path.name
    if name in exclude_dirs:
        return True
    if _match_name(name, exclude_dir_patterns):
        return True
    if _match_path(rel_path, exclude_paths):
        return True
    return False


def _should_skip_file(
    rel_path: Path,
    exclude_file_patterns: list[str],
    exclude_paths: list[str],
) -> bool:
    if rel_path.parts and rel_path.parts[0].lower() == "data":
        if rel_path.suffix.lower() != ".parquet":
            return True
    parts = [part.lower() for part in rel_path.parts]
    if len(parts) >= 2 and parts[0] == "hftbacktest" and parts[1] == "examples":
        if rel_path.suffix.lower() == ".gz":
            return True
    if _match_name(rel_path.name, exclude_file_patterns):
        return True
    if _match_path(rel_path, exclude_paths):
        return True
    return False


def make_archive(root: Path, out_path: Path, exclude_paths: list[str]) -> None:
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    exclude_dir_patterns = list(DEFAULT_EXCLUDE_DIR_PATTERNS)
    exclude_file_patterns = list(DEFAULT_EXCLUDE_FILE_PATTERNS)

    root = root.resolve()
    out_path = out_path.resolve()

    if out_path.exists():
        out_path.unlink()

    files_added = 0
    bytes_added = 0

    print(f"archiving_root={root}")
    print(f"archive_path={out_path}")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).resolve().relative_to(root)
            dirnames[:] = [
                d
                for d in dirnames
                if not _should_skip_dir(
                    rel_dir / d,
                    exclude_dirs,
                    exclude_dir_patterns,
                    exclude_paths,
                )
            ]

            for filename in filenames:
                full_path = Path(dirpath) / filename
                if full_path.resolve() == out_path:
                    continue
                rel_path = full_path.resolve().relative_to(root)
                if _should_skip_file(rel_path, exclude_file_patterns, exclude_paths):
                    continue
                zf.write(full_path, rel_path.as_posix())
                files_added += 1
                try:
                    bytes_added += full_path.stat().st_size
                except OSError:
                    pass

    print(f"files_added={files_added} bytes_added={bytes_added}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a zip archive of the project.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional glob patterns to exclude (relative paths).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.output:
        out_path = Path(args.output)
    else:
        resolved_root = root.resolve()
        out_path = resolved_root / (resolved_root.name + ".zip")

    make_archive(root, out_path, args.exclude)


if __name__ == "__main__":
    main()
