"""Validate full book coverage, statuses, generated indexes, and local links."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import jsonschema

from build_catalog import ROOT, generated_files


def validate(catalog: dict | None = None) -> dict:
    catalog = catalog or json.loads((ROOT / "coverage/catalog.json").read_text())
    schema = json.loads((ROOT / "coverage/catalog.schema.json").read_text())
    jsonschema.validate(catalog, schema)
    entries = catalog["entries"]
    keys = [e["key"] for e in entries]
    assert len(keys) == len(set(keys)), "Duplicate catalog key"
    category_ids = {c["id"] for c in catalog["categories"]}
    assert category_ids == {"examples", "computational_exercises", "other_exercises"}
    for c in catalog["categories"]:
        assert c["count"] == sum(e["category"] == c["id"] for e in entries)
        assert (ROOT / c["index_path"]).is_file()

    audit = json.loads((ROOT / "coverage/source_audit.json").read_text())
    source = {e["id"]: e for e in audit["exercises"]}
    exercises = [e for e in entries if e["kind"] == "exercise"]
    assert len(exercises) == len(source) == 337
    assert {e["id"] for e in exercises} == set(source), "Numbering gap or unexpected exercise"
    computed = [e for e in exercises if e["category"] == "computational_exercises"]
    pending = [e for e in exercises if e["category"] == "other_exercises"]
    assert len(computed) == 16 and len(pending) == 321
    assert all(e["status"] == "verified" for e in computed), "Computational completion is not verified"
    assert {e["id"] for e in computed} == {e["id"] for e in source.values() if e["selected_for_implementation"]}
    assert sum(e.get("textbook_role") == "explicit_computation" for e in computed) == 11
    assert sum(e.get("textbook_role") == "numerical_extension" for e in computed) == 5
    for e in exercises:
        assert e["key"] == f"exercise:{e['id']}"
        assert e["chapter"] == int(e["id"].split(".")[0])
        assert e["source_url"] == source[e["id"]]["source_url"]
        assert e["source_page"] == source[e["id"]]["source_page"]
        assert e["source_pdf_page"] == int(e["source_url"].split("#page=")[1])
    for e in pending:
        assert e["status"] == "pending"
        assert not any(k in e for k in ("solution_path", "results_path", "implementation_module")), "Pending exercise claims delivered files"
        assert not (ROOT / "docs/solutions" / f"{e['id']}.md").exists(), "Unexpected pending solution stub"

    for e in entries:
        if e["kind"] == "worked_example":
            assert e["category"] == "examples"
            assert e["key"] == f"example:{e['id']}"
            if e["status"] == "verified":
                assert e.get("output_match_verified") is True, f"Unverified worked-example output: {e['key']}"
        if e["status"] == "verified":
            assert "solution_path" in e and "results_path" in e, f"Missing evidence: {e['key']}"
            for key in ("solution_path", "results_path", "code_path", "verification_path"):
                if key in e:
                    assert (ROOT / e[key]).is_file(), f"Missing file: {e[key]}"
        elif e["status"] == "pending":
            assert "solution_path" not in e and "results_path" not in e, f"Pending item claims output: {e['key']}"
    assert catalog["summary"]["numbered_exercises"] == 337
    assert catalog["summary"]["computational_exercises_verified"] == len(computed)
    assert catalog["summary"]["other_exercises_pending"] == len(pending)
    examples = [e for e in entries if e["category"] == "examples"]
    assert catalog["summary"]["worked_examples_catalogued"] == len(examples)
    assert catalog["summary"]["worked_examples_reproduced"] == sum(e["status"] == "verified" for e in examples)
    for chapter in catalog["chapters"]:
        ch_entries = [e for e in exercises if e["chapter"] == chapter["chapter"]]
        assert chapter["numbered_exercises"] == len(ch_entries)
        assert chapter["other_exercises"] == sum(e["category"] == "other_exercises" for e in ch_entries)
        assert chapter["computational_exercises"] == sum(e["category"] == "computational_exercises" for e in ch_entries)
    for relative, text in generated_files().items():
        assert (ROOT / relative).read_text() == text, f"Stale generated catalog: {relative}"
    check_local_links()
    return catalog["summary"]


def heading_anchors(path: Path) -> set[str]:
    anchors = set()
    counts = {}
    for heading in re.findall(r"^#{1,6} (.+)$", path.read_text(), re.M):
        heading = re.sub(r"[\[\]`*]", "", heading).lower()
        base = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = counts.get(base, 0)
        anchors.add(base if count == 0 else f"{base}-{count}")
        counts[base] = count + 1
    return anchors


def check_local_links() -> None:
    pages = [ROOT / "README.md", ROOT / "coverage/README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for page in pages:
        for target in re.findall(r"\]\(([^)\s]+)\)", page.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            path = (page.parent / unquote(parsed.path)).resolve() if parsed.path else page
            assert path.is_relative_to(ROOT), f"Link outside repository: {page}: {target}"
            assert path.exists(), f"Broken local link: {page}: {target}"
            if parsed.fragment and path.suffix == ".md":
                assert unquote(parsed.fragment) in heading_anchors(path), f"Missing heading: {page}: {target}"


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
