"""Source integrity and annotation-boundary tests; never run model code."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.verify_source_snippets import ROOT, validate


class SourceSnippetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in (
            "content/source-snippets.json",
            "research/source-evidence.json",
            "source.lock.json",
        ):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.path = self.root / "content/source-snippets.json"
        self.data = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_all_reference_entries_have_valid_verbatim_archives(self):
        result = validate(self.root)
        self.assertEqual(result["sources"], 29)
        self.assertEqual(result["excerpts"], 54)

    def test_code_changes_are_rejected(self):
        self.data["entries"][0]["excerpts"][0]["code"] += "# injected note\n"
        self.save()
        with self.assertRaisesRegex(ValueError, "Line count|checksum"):
            validate(self.root)

    def test_annotations_must_target_real_excerpt_lines(self):
        self.data["entries"][0]["excerpts"][0]["annotations"][0]["line"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "Invalid annotation"):
            validate(self.root)

    def test_wrong_commit_is_rejected(self):
        self.data["entries"][0]["commit"] = "0" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "Commit mismatch"):
            validate(self.root)

    def test_cached_source_must_match_git_blob(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "B-Q2.txt").write_text("not the pinned source\n")
        with self.assertRaisesRegex(ValueError, "Upstream blob mismatch"):
            validate(self.root, cache)

    def test_wrong_line_offsets_are_rejected_even_with_matching_excerpt_checksum(self):
        # A synthetic single-file fixture isolates the exact extraction boundary check.
        entry = self.data["entries"][0]
        raw = b"first\nsecond\nthird\n"
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        entry["git_blob_sha"] = blob
        entry["excerpts"] = [
            {
                "start_line": 2,
                "end_line": 2,
                "code": "first\n",
                "sha256": hashlib.sha256(b"first\n").hexdigest(),
                "annotations": [{"line": 2, "text": "fixture only"}],
            }
        ]
        self.data["entries"] = [entry]
        self.save()
        evidence_path = self.root / "research/source-evidence.json"
        evidence = json.loads(evidence_path.read_text())["entries"][0]
        evidence["git_blob_sha"] = blob
        evidence_path.write_text(json.dumps({"entries": [evidence]}))
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "B-Q2.txt").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "differs from upstream lines"):
            validate(self.root, cache)
