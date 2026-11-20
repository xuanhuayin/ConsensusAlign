import json
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUTHORS = ["Xuanhua Yin", "Shunqi Mao", "Chuanzhi Xu", "Boyu Wei", "Wei Guo", "Weidong Cai"]


def test_public_authorship_and_software_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert [author["name"] for author in project["authors"]] == AUTHORS
    assert [author["given-names"] + " " + author["family-names"] for author in citation["authors"]] == AUTHORS
    assert all(author["affiliation"] == "University of Sydney" for author in citation["authors"])
    assert project["version"] == citation["version"]
    assert citation["preferred-citation"]["type"] == "unpublished"
    assert "url" not in citation and "doi" not in citation


def test_local_documentation_links_and_overview_exist():
    for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            target = link.split("#", 1)[0]
            assert (path.parent / target).exists(), (path.name, target)
    assert (ROOT / "docs/assets/overview.svg").is_file()


def test_reference_results_have_explicit_provenance():
    values = json.loads((ROOT / "benchmarks/reference/coco_base.json").read_text())
    assert values["status"] == "paper reference values, not a new run of this repository"
    assert values["failure_threshold"] == 0.05
    assert len(values["records"]) == 2


def test_actions_are_pinned_and_read_only():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    for ref in re.findall(r"uses: ([^\s]+)", workflow):
        assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", ref)
