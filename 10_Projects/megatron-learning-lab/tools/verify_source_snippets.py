"""Validate curated, verbatim source excerpts; optional fixed-commit cache fetch."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def validate(root: Path = ROOT, cache: Path | None = None) -> dict:
    evidence = {
        e["id"]: e
        for e in json.loads((root / "research/source-evidence.json").read_text())[
            "entries"
        ]
    }
    lock = json.loads((root / "source.lock.json").read_text())["repositories"]
    entries = json.loads((root / "content/source-snippets.json").read_text())["entries"]
    seen = set()
    count = 0
    for entry in entries:
        source_id = entry["source_id"]
        if source_id in seen or source_id not in evidence:
            raise ValueError(f"Duplicate or unknown source: {source_id}")
        seen.add(source_id)
        source = evidence[source_id]
        for key in ("path", "git_blob_sha"):
            if entry[key] != source[key]:
                raise ValueError(f"Lineage mismatch: {source_id}/{key}")
        if entry["commit"] != lock[source["repo_key"]]["commit"]:
            raise ValueError(f"Commit mismatch: {source_id}")
        lines = None
        if cache is not None:
            raw = (cache / f"{source_id}.txt").read_bytes()
            blob = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            if blob != entry["git_blob_sha"]:
                raise ValueError(f"Upstream blob mismatch: {source_id}")
            lines = raw.decode("utf-8").splitlines(keepends=True)
        if not entry["excerpts"]:
            raise ValueError(f"Empty excerpt collection: {source_id}")
        for excerpt in entry["excerpts"]:
            a, b, code = excerpt["start_line"], excerpt["end_line"], excerpt["code"]
            if not (isinstance(a, int) and isinstance(b, int) and 1 <= a <= b):
                raise ValueError(f"Invalid line range: {source_id}")
            if len(code.splitlines()) != b - a + 1:
                raise ValueError(f"Line count mismatch: {source_id}")
            if hashlib.sha256(code.encode()).hexdigest() != excerpt["sha256"]:
                raise ValueError(f"Excerpt checksum mismatch: {source_id}")
            if lines is not None and (
                b > len(lines) or "".join(lines[a - 1 : b]) != code
            ):
                raise ValueError(f"Excerpt differs from upstream lines: {source_id}")
            if not excerpt["annotations"] or any(
                not a <= note["line"] <= b or not note["text"].strip()
                for note in excerpt["annotations"]
            ):
                raise ValueError(f"Invalid annotation: {source_id}")
            count += 1
    if seen != set(evidence):
        raise ValueError("Source coverage incomplete")
    return {
        "sources": len(seen),
        "excerpts": count,
        "checks": "full_blob_and_verbatim_lines"
        if cache
        else "offline_lineage_and_excerpt_integrity",
    }


def fetch_cache(root: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for source in json.loads((root / "research/source-evidence.json").read_text())[
        "entries"
    ]:
        url = (
            source["url"]
            .replace("https://github.com/", "https://raw.githubusercontent.com/")
            .replace("/blob/", "/")
        )
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != source["git_blob_sha"]:
            raise ValueError(f"Refusing changed source: {source['id']}")
        (cache / f"{source['id']}.txt").write_bytes(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, help="Verify against cached complete source files"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Explicitly download fixed references into --cache",
    )
    args = parser.parse_args()
    if args.fetch and args.cache is None:
        parser.error("--fetch requires --cache")
    if args.fetch:
        fetch_cache(ROOT, args.cache)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(json.dumps(validate(cache=args.cache), ensure_ascii=False))
