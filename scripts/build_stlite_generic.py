#!/usr/bin/env python3
"""Package ANY (multipage) Streamlit app into one self-contained stlite HTML
file that runs entirely in the browser via Pyodide -- no server, no build
step beyond running this script.

Only needs the Python 3 standard library -- no pip installs, no internet to
*generate* the file. The generated HTML itself needs internet when *opened*
in a browser, to fetch stlite/Pyodide/the Python packages from a CDN at
runtime -- that's unrelated to generating it.

Zero-config usage (auto-detects entrypoint + requirements, embeds no data
files unless told to):
    python3 build_stlite_generic.py --root /path/to/app

Typical usage with data files:
    python3 build_stlite_generic.py --root /path/to/app \
        --data data --data "assets/*.png" \
        --requirements streamlit,pandas,numpy

Reusable config instead of long command lines (CLI flags override the file):
    python3 build_stlite_generic.py --root /path/to/app --config stlite.config.json

stlite.config.json:
    {
      "entrypoint": "app.py",
      "requirements": ["streamlit", "pandas"],
      "data": ["data/*.csv", "assets"],
      "exclude": ["tests/*"],
      "title": "My App"
    }

Caveat: this embeds whatever your --data patterns match, with no idea which
files your app's code actually reads. For an app whose data directory has
far more in it than the code uses (e.g. a raw export with mixed/ambiguous
filenames), write a small app-specific script instead of relying on globs
-- see scripts/build_stlite.py in this repo for a worked example that
disambiguates files by content, not filename.
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import sys
from pathlib import Path

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", ".env", "node_modules", "build", "dist",
    ".idea", ".vscode", ".ruff_cache", ".tox",
}
DEFAULT_ENTRYPOINTS = ["app.py", "streamlit_app.py", "Home.py", "main.py"]
WARN_SIZE_MB = 25


def in_excluded_dir(rel: Path) -> bool:
    return any(part in DEFAULT_EXCLUDE_DIRS for part in rel.parts)


def matches_any(rel_str: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_str, pat) for pat in patterns)


def find_entrypoint(root: Path) -> Path:
    for name in DEFAULT_ENTRYPOINTS:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "Couldn't auto-detect an entrypoint (looked for "
        f"{', '.join(DEFAULT_ENTRYPOINTS)}) under {root}. Pass --entrypoint."
    )


def collect_py_files(root: Path, exclude_globs: list[str], never: set[Path]) -> list[Path]:
    out = []
    for p in root.rglob("*.py"):
        if not p.is_file() or p.resolve() in never:
            continue
        rel = p.relative_to(root)
        if in_excluded_dir(rel) or matches_any(rel.as_posix(), exclude_globs):
            continue
        out.append(p)
    return sorted(out)


def collect_data_files(root: Path, patterns: list[str], exclude_globs: list[str], never: set[Path]) -> list[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for p in root.glob(pattern):
            targets = p.rglob("*") if p.is_dir() else [p]
            for f in targets:
                if not f.is_file():
                    continue
                resolved = f.resolve()
                if resolved in never or resolved in seen:
                    continue
                rel = f.relative_to(root)
                if in_excluded_dir(rel) or matches_any(rel.as_posix(), exclude_globs):
                    continue
                seen.add(resolved)
    return sorted(seen)


def detect_requirements(root: Path) -> list[str]:
    req_txt = root / "requirements.txt"
    if req_txt.is_file():
        reqs = [
            line.strip()
            for line in req_txt.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith(("#", "-r", "-e", "--"))
        ]
        if reqs:
            return reqs

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib
        except ModuleNotFoundError:
            tomllib = None
        if tomllib is not None:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            deps = data.get("project", {}).get("dependencies", [])
            if deps:
                # Trim environment markers / extras down to "name<specifier>",
                # since Poetry-style "(>=1.2)" confuses micropip.
                return [dep.split(";")[0].strip() for dep in deps]

    print(
        "WARNING: couldn't auto-detect requirements (no requirements.txt or "
        "pyproject.toml [project.dependencies] found); defaulting to "
        "['streamlit']. Pass --requirements to override.",
        file=sys.stderr,
    )
    return ["streamlit"]


def read_for_embedding(path: Path) -> tuple[object, bool]:
    """Text files are embedded as UTF-8 strings. Anything that fails strict
    UTF-8 decoding is treated as binary and base64-encoded; the page decodes
    it back to raw bytes before handing it to stlite."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return base64.b64encode(raw).decode("ascii"), True


def load_config(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = load_config(Path(args.config).resolve() if args.config else root / "stlite.config.json" if (root / "stlite.config.json").is_file() else None)

    entrypoint_name = args.entrypoint or config.get("entrypoint")
    entrypoint = (root / entrypoint_name) if entrypoint_name else find_entrypoint(root)
    if not entrypoint.is_file():
        raise SystemExit(f"Entrypoint not found: {entrypoint}")
    entrypoint_vp = entrypoint.relative_to(root).as_posix()

    out_path = Path(args.out) if args.out else root / f"{entrypoint.stem}_stlite.html"
    this_script = Path(__file__).resolve()
    never_embed = {this_script, out_path.resolve()}

    exclude_globs = list(config.get("exclude", [])) + list(args.exclude or [])

    if args.requirements:
        requirements = [r.strip() for r in args.requirements.split(",") if r.strip()]
    elif config.get("requirements"):
        requirements = list(config["requirements"])
    else:
        requirements = detect_requirements(root)

    data_patterns = list(args.data or config.get("data", []))
    if not data_patterns:
        print(
            "No --data patterns (and none in stlite.config.json) -- embedding "
            "Python sources only, no data files. Pass --data to include some, "
            "e.g. --data data --data \"assets/*.png\".",
            file=sys.stderr,
        )

    py_files = collect_py_files(root, exclude_globs, never_embed)
    data_files = collect_data_files(root, data_patterns, exclude_globs, never_embed)

    files: dict[str, object] = {}
    total_bytes = 0
    for p in py_files + data_files:
        content, is_binary = read_for_embedding(p)
        vp = p.relative_to(root).as_posix()
        files[vp] = {"base64": content} if is_binary else content
        size = p.stat().st_size
        total_bytes += size
        print(f"embedded {vp} ({size:,} bytes{', binary' if is_binary else ''})")

    cfg = {"entrypoint": entrypoint_vp, "requirements": requirements, "files": files}
    config_json = json.dumps(cfg).replace("</", "<\\/")

    stlite_version = args.stlite_version
    title = args.title or config.get("title") or entrypoint.stem.replace("_", " ").title()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{stlite_version}/build/stlite.css" />
<style>html,body{{height:100%;margin:0}}#root{{height:100%}}</style>
</head>
<body>
<div id="root"></div>
<script id="app-data" type="application/json">
{config_json}
</script>
<script type="module">
  import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{stlite_version}/build/stlite.js";
  const raw = JSON.parse(document.getElementById("app-data").textContent);
  for (const [path, value] of Object.entries(raw.files)) {{
    if (value && typeof value === "object" && "base64" in value) {{
      const binary = atob(value.base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      raw.files[path] = bytes;
    }}
  }}
  mount(raw, document.getElementById("root"));
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDONE: {out_path} ({size_mb:.2f} MB, {len(files)} files, {total_bytes:,} raw bytes)")
    if size_mb > WARN_SIZE_MB:
        print(
            f"WARNING: output is over {WARN_SIZE_MB} MB -- consider narrowing "
            "--data patterns if that's more than you meant to embed.",
            file=sys.stderr,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="Project root (default: current directory)")
    p.add_argument("--config", help="Path to a JSON config file (default: stlite.config.json in --root, if present)")
    p.add_argument("--entrypoint", help="Main app file, relative to --root (default: auto-detect app.py / streamlit_app.py / Home.py / main.py)")
    p.add_argument("--out", help="Output HTML path (default: <entrypoint-stem>_stlite.html in --root)")
    p.add_argument("--requirements", help="Comma-separated pip package list (default: auto-detect from requirements.txt or pyproject.toml)")
    p.add_argument("--data", action="append", help="Glob pattern (relative to --root) for data files/dirs to embed; repeatable. None by default.")
    p.add_argument("--exclude", action="append", help="Glob pattern (relative to --root) to exclude from both --data and the .py scan; repeatable.")
    p.add_argument("--stlite-version", default="0.85.1", help="@stlite/browser version to load from jsDelivr")
    p.add_argument("--title", help="HTML <title> (default: derived from entrypoint filename)")
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
