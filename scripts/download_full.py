"""Download and pin the official full release used by the mini experiment."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from download_mini import ROOT, REPO, REVISION, fetch


def main():
    metadata = json.loads(fetch("https://huggingface.co/api/datasets/" + REPO + "/revision/" + REVISION))
    paths = sorted(s["rfilename"] for s in metadata["siblings"]
                   if s["rfilename"].startswith("data/full/") and s["rfilename"].endswith(".json"))
    assert len(paths) == 1140, "Expected official 1,140-trajectory full split"

    def download(path):
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = target.read_bytes() if target.exists() else fetch(
            "https://huggingface.co/datasets/" + REPO + "/resolve/" + REVISION + "/" + path)
        row = json.loads(raw)
        assert isinstance(row.get("history"), list)
        target.write_bytes(raw)
        return {"path": path, "question_ID": row["question_ID"],
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for entry in pool.map(download, paths):
            entries.append(entry)
            if len(entries) % 50 == 0:
                print("Downloaded %d/1140" % len(entries), flush=True)
    manifest = {"repository": REPO, "revision": REVISION, "subset": "default", "split": "test", "files": entries}
    (ROOT / "data" / "full_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    mini = json.loads((ROOT / "data/manifest.json").read_text())
    by_id = {e["question_ID"]: e for e in entries}
    for entry in mini["files"]:
        assert entry["sha256"] == by_id[entry["question_ID"]]["sha256"], "Mini/full content differs"
    print("Full manifest saved. All 200 mini files exactly match full counterparts.")


if __name__ == "__main__":
    main()
