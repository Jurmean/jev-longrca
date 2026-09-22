"""Launch one resumable Full shard per explicitly selected idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/laya_full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", help="Physical GPU indices, e.g. 0 or 0,1. Default: one idle GPU; resume keeps the saved mapping.")
    args = parser.parse_args()
    meta_path = OUT / "launcher.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise SystemExit("Unset CUDA_VISIBLE_DEVICES for this launcher and select physical GPUs with --gpus instead.")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    idle = [parts[0] for line in raw.splitlines()
            for parts in [[int(x.strip()) for x in line.split(",")]]
            if parts[1] < 256 and parts[2] == 0]
    try:
        gpus = [int(x.strip()) for x in args.gpus.split(",")] if args.gpus else old.get("gpus", idle[:1])
    except ValueError:
        raise SystemExit("--gpus must contain comma-separated integer GPU indices")
    if not gpus:
        raise SystemExit("No idle GPU available")
    if len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise SystemExit("GPU indices must be unique and nonnegative")
    if any(g not in idle for g in gpus):
        raise SystemExit("At least one selected GPU is occupied or unavailable; no workers started")
    OUT.mkdir(parents=True, exist_ok=True)
    if old and old["gpus"] != gpus:
        raise SystemExit("Resume with the original GPU/shard mapping; do not silently repartition")
    meta = dict(started_unix=old.get("started_unix", time.time()), gpus=gpus,
                attempts=old.get("attempts", []) + [{"started_unix": time.time()}])
    meta_path.write_text(json.dumps(meta, indent=2))
    processes = []
    for shard, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        log = (OUT / ("worker_%02d.log" % shard)).open("a")
        p = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/evaluate_laya.py"),
                              "--shard", str(shard), "--shards", str(len(gpus))],
                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        processes.append(p)
        print("Started shard %d/%d on GPU %d, PID %d" % (shard, len(gpus), gpu, p.pid), flush=True)
    previous = -1
    while any(p.poll() is None for p in processes):
        n = len(list((OUT / "predictions").glob("*.json")))
        if n != previous:
            print("Completed %d/1140; wall %.1f minutes" % (n, (time.time()-meta["started_unix"])/60), flush=True)
            previous = n
        time.sleep(10)
    meta.update(finished_unix=time.time(), exit_codes=[p.returncode for p in processes])
    meta["attempts"][-1]["finished_unix"] = meta["finished_unix"]
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    if any(meta["exit_codes"]):
        raise SystemExit("At least one worker failed; inspect worker logs before resuming")


if __name__ == "__main__":
    main()
