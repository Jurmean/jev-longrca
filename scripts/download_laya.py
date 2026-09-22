"""Download the exact audited Laya artifacts; Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "convaiinnovations/laya"
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local files without accessing the network")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reports/laya_download_manifest.json").read_text())
    if manifest["repo"] != REPO or manifest["revision"] != REVISION:
        raise SystemExit("Unexpected model revision in the pinned manifest")
    model_root = ROOT / "models/laya"
    for name, entry in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe model artifact path")
        target = model_root / relative
        if matches(target, entry):
            print("Verified " + name, flush=True)
            continue
        if args.verify_only:
            raise SystemExit("Missing or checksum mismatch: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".download")
        for attempt in range(4):
            try:
                url = "https://huggingface.co/%s/resolve/%s/%s" % (REPO, REVISION, name)
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        out.write(chunk)
                if not matches(partial, entry):
                    raise ValueError("Downloaded artifact checksum mismatch")
                partial.replace(target)
                print("Downloaded and verified " + name, flush=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    print("Pinned Laya artifacts are ready; model inference can now run offline.")


if __name__ == "__main__":
    main()
