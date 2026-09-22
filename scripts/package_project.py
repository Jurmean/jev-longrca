"""Create an allowlisted source bundle, without credentials or execution caches."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    "phase1_report.md", "phase1_metrics.json", "phase1_predictions.csv", "result_audit.json",
    "full_report.md", "full_metrics.json", "full_predictions.csv", "full_result_audit.json", "full_timing_audit.json",
    "laya_full_report.md", "laya_full_metrics.json", "laya_full_predictions.csv", "laya_result_audit.json",
    "laya_download_manifest.json", "laya_environment.txt", "laya_diagnostics.json", "laya_sanity_checks.json",
]
SECRET_PATTERNS = [re.compile(rb"apikey_[A-Za-z0-9_-]{24,}"),
                   re.compile(rb"jv_live_[A-Za-z0-9_-]{24,}"),
                   re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
                   re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]


def selected_paths(root):
    names = ["README.md", "REPRODUCE.md", ".gitignore", ".env.example", "requirements.txt",
             "requirements-jev.txt", "requirements-lock-cu124.txt",
             "data/manifest.json", "data/full_manifest.json", "data/DATASET_CARD.md",
             "results/phase1/config.json", "results/full/config.json"]
    names += ["reports/" + name for name in REPORTS]
    paths = [root / name for name in names]
    paths += sorted((root / "scripts").glob("*.py"))
    paths += sorted((root / "tests").glob("*.py"))
    return paths


def private_values(root):
    """Only compare private values in memory; never print them or add .env to the archive."""
    values = []
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if not line.strip().startswith("#") and "=" in line:
                value = line.split("=", 1)[1].strip().strip("\"'")
                if len(value) >= 16:
                    values.append(value.encode())
    return values


def reject_secrets(raw, name, values):
    if any(value in raw for value in values) or any(p.search(raw) for p in SECRET_PATTERNS):
        raise ValueError("Possible credential in selected file; bundle not written: " + name)


def main():
    payloads = {}
    values = private_values(ROOT)
    for path in selected_paths(ROOT):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected a regular selected file: " + str(path.relative_to(ROOT)))
        name = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        reject_secrets(raw, name, values)
        payloads[name] = raw
    manifest = {"format": 1, "purpose": "Source, pinned dependencies and reference results; fresh inference required.",
                "excludes": ["private .env", "virtual environments", "model weights and downloaded upstream code",
                             "dataset trajectory JSON", "per-case prediction caches", "raw API/local call archives", "logs", ".git"],
                "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                          for name, raw in sorted(payloads.items())}}
    payloads["SHARE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    target = out / "Jev-longRCA-source.zip"
    partial = out / "Jev-longRCA-source.zip.tmp"
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(payloads.items()):
            archive.writestr("Jev-longRCA/" + name, raw)
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        for name, raw in payloads.items():
            assert archive.read("Jev-longRCA/" + name) == raw
    partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(digest + "  " + target.name + "\n")
    print("Created %s: %d files, %.1f KiB; allowlist, credential scan and archive checks passed."
          % (target, len(payloads), target.stat().st_size / 1024))


if __name__ == "__main__":
    main()
