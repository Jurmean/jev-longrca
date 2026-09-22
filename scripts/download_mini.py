"""Fetch the official fixed mini split without third-party dependencies."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "CLoud5-real/longrca-bench"
REVISION = "9f45acb66948d5d20c663b4ce4ec8ea5ab0076dd"


def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    metadata = json.loads(fetch("https://huggingface.co/api/datasets/" + REPO + "/revision/" + REVISION))
    paths = sorted(s["rfilename"] for s in metadata["siblings"]
                   if s["rfilename"].startswith("data/mini/") and s["rfilename"].endswith(".json"))
    if len(paths) != 200:
        raise ValueError("Official mini split must contain 200 files")

    def download(path):
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = target.read_bytes() if target.exists() else fetch(
            "https://huggingface.co/datasets/" + REPO + "/resolve/" + REVISION + "/" + path)
        row = json.loads(raw)
        if not isinstance(row.get("history"), list):
            raise ValueError("Invalid trajectory: " + path)
        target.write_bytes(raw)
        return {"path": path, "question_ID": row["question_ID"],
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for entry in pool.map(download, paths):
            entries.append(entry)
            if len(entries) % 20 == 0:
                print("Downloaded %d/200" % len(entries), flush=True)
    manifest = {"repository": REPO, "revision": REVISION, "subset": "mini", "split": "test", "files": entries}
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "data" / "DATASET_CARD.md").write_bytes(fetch(
        "https://huggingface.co/datasets/" + REPO + "/resolve/" + REVISION + "/README.md"))
    print("Pinned dataset manifest saved.")


if __name__ == "__main__":
    main()
