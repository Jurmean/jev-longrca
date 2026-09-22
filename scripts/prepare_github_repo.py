"""Build an allowlisted, credential-scanned repository without changing frozen runs."""
import hashlib
import json
from pathlib import Path

import package_project

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dist/jev-longrca-github"


def main():
    if DEST.exists():
        raise SystemExit("Destination already exists; preserve it or choose a new export location")
    values = []
    for path in ROOT.glob(".env*"):
        if not path.is_file() or path.name.endswith("example"):
            continue
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if len(value) >= 12:
                    values.append(value.encode())
    payloads = {}
    for path in package_project.selected_paths(ROOT):
        if path.name == "README.md":
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError("Expected regular source file: " + str(path.relative_to(ROOT)))
        payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    notes = ROOT / "docs/EXPERIMENT_NOTES.md"
    if not notes.exists():
        notes = ROOT / "README.md"
    payloads["docs/EXPERIMENT_NOTES.md"] = notes.read_text().replace(
        "](REPRODUCE.md)", "](../REPRODUCE.md)").encode()
    for path in sorted((ROOT / "figures").glob("*")):
        if path.is_file() and not path.is_symlink():
            payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("jev_rcta_feasibility.md", "jev_rcta_hosted_full_report.md", "jev_rcta_hosted_audit.json",
                 "jev_rcta_hosted_predictions.csv", "jev_error_analysis.md", "jev_error_analysis.json",
                 "jev_error_analysis.csv"):
        payloads["reports/" + name] = (ROOT / "reports" / name).read_bytes()
    metrics = json.loads((ROOT / "reports/jev_rcta_hosted_metrics.json").read_text())
    metrics.pop("details", None)
    metrics["export_note"] = "Aggregate-only export: raw evidence cards and per-case embedded trajectory excerpts omitted. See prediction CSV."
    payloads["reports/jev_rcta_hosted_metrics.json"] = (json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode()
    provenance = ROOT / "results/jev_rcta_hosted_full_v2"
    if not provenance.exists():
        provenance = ROOT / "reproducibility/jev-rcta-v2"
    for name in ("config.json", "selection.json", "data_audit.json", "transport_recovery.json"):
        payloads["reproducibility/jev-rcta-v2/" + name] = (provenance / name).read_bytes()
    payloads[".env.jev-hosted.example"] = b"# Copy to .env.jev-hosted or set this environment variable.\nJEV_HOSTED_API_KEY=replace_with_your_own_key\n"
    payloads[".gitignore"] = b""".env
.env.*
!.env.example
!.env.jev-hosted.example
__pycache__/
*.py[cod]
.venv*/
.python-runtime/
.agents/
.codex/
models/
data/mini/
data/full/
dist/
*.log
results/*
!results/phase1/
results/phase1/*
!results/phase1/config.json
!results/full/
results/full/*
!results/full/config.json
"""
    readme = ROOT / "github_readme.md"
    if not readme.exists():
        readme = ROOT / "README.md"
    payloads["README.md"] = readme.read_bytes()
    payloads[".github/workflows/tests.yml"] = b"""name: Offline tests
on: [push, pull_request, workflow_dispatch]
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Run offline tests without API credentials
        run: python -m unittest discover -s tests -v
"""
    for name, raw in payloads.items():
        package_project.reject_secrets(raw, name, values)
        if any(part in ("models", ".venv", "calls", "predictions") for part in Path(name).parts):
            raise ValueError("Private or bulky payload selected: " + name)
    manifest = {"format": 1, "purpose": "GitHub source repository export",
                "omitted": ["private environment files", "raw trajectory data", "model weights", "virtual environments",
                            "request/response caches", "per-case evidence excerpts", "runtime logs"],
                "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                          for name, raw in sorted(payloads.items())}}
    payloads["REPOSITORY_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    for name, raw in payloads.items():
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    print(json.dumps({"directory": str(DEST), "files": len(payloads), "bytes": sum(map(len, payloads.values())),
                      "credentials_scan": "passed"}, indent=2))


if __name__ == "__main__":
    main()
