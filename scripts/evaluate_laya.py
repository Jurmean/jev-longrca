"""Offline Laya Full evaluation with explicit, audited token budgets.

Only trajectory history enters predict(). Gold labels are used after inference.
The upstream checkpoint/source and the existing JEV runner remain unchanged.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

import evaluate as ev
from evaluate_full import load_full

ROOT = ev.ROOT
MODEL_DIR = ROOT / "models/laya"
sys.path.insert(0, str(MODEL_DIR))
CONFIG = dict(ev.CONFIG, protocol="laya-choice-recall-token-budget-v1",
              model="convaiinnovations/laya", model_revision="1c5edc17a7acd8701df6fc341c0d179f1c62c982",
              max_len=8192, max_recall_options=16, shared_token_budget=2048,
              task_bytes=2000, outcome_bytes=4000,
              checkpoint="English root; no language routing or fine-tuning",
              dtype="float32 weights; bfloat16 CUDA autocast")
CONFIG.pop("chunk_bytes")
CONFIG.pop("piece_bytes")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class Budget:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._headers = {}

    def tokens(self, text):
        return self.tok(text.replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"]

    def state_tokens(self, state):
        # Match upstream serialize_state exactly (including its default spaces).
        return self.tokens(json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state)

    def header(self, q):
        cache_key = ev.dumps(q)
        if cache_key in self._headers:
            return self._headers[cache_key]
        from rl_common import render_options
        ins = self.tokens(q["type"] + " question: " + q["instructions"])
        opts = render_options({"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]})
        lengths = [len(self.tokens(" " + opt)) for opt in opts]
        if any(n > 48 for n in lengths):
            raise ValueError("Upstream option text would be truncated at 48 tokens")
        length = len(ins) + sum(n + 1 for n in lengths)
        self._headers[cache_key] = length
        return length

    def fits(self, state, questions):
        n = len(self.state_tokens(state))
        return all(n + self.header(q) + 4 <= CONFIG["max_len"] for q in questions.values())


class LocalClient:
    def __init__(self, agent, directory):
        self.agent = agent
        self.budget = Budget(agent.tok)
        self.directory = directory
        self.calls = []
        directory.mkdir(parents=True, exist_ok=True)

    def call(self, state, questions, tag):
        from rl_common import build_sequence
        payload = {"model": CONFIG["model"], "state": state, "questions": questions}
        digest = ev.sha(ev.dumps(payload).encode())
        path = self.directory / (tag + ".json.gz")
        if path.exists():
            with gzip.open(path, "rt") as f:
                saved = json.load(f)
            if saved["request_sha256"] != digest:
                raise ValueError("Cached local request differs")
            self.calls.append(saved)
            return saved["response"]
        started = time.monotonic()
        answers, audits, n_tokens = {}, {}, 0
        state_tokens = self.budget.state_tokens(state)
        for qid, q in questions.items():
            header = self.budget.header(q)
            expected = len(state_tokens) + header + 4
            if expected > CONFIG["max_len"]:
                raise ValueError("Refusing silent context truncation: %d tokens" % expected)
            self.agent.cfg["max_len"] = CONFIG["max_len"]
            self.agent.cfg["head_max_len"] = max(header, 16 + header)
            seq, markers = build_sequence(self.agent.tok, state, self.agent._to_internal(q),
                                          CONFIG["max_len"], self.agent.cfg["head_max_len"])
            if len(seq) != expected or len(markers) != len(q["criteria"]):
                raise ValueError("Upstream sequence construction truncated input")
            if seq[-len(state_tokens)-1:-1] != state_tokens:
                raise ValueError("Model does not see the complete supplied state")
            if len(q["criteria"]) == 1:
                # The upstream act head calls topk(2). A forced singleton needs no forward pass.
                key = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": key, "probabilities": {key: 1.0},
                                "confidence": 1.0, "deterministic_singleton": True}
                actual = 0
            else:
                response = self.agent.system_one(state, {qid: q})
                actual = response["usage"]["input_tokens"]
                if actual != expected:
                    raise ValueError("Runtime token count differs from audit")
                answers[qid] = response["answers"][qid]
                if not all(math.isfinite(p) for p in answers[qid]["probabilities"].values()):
                    raise ValueError("Non-finite model probabilities")
            n_tokens += actual
            audits[qid] = {"state_tokens": len(state_tokens), "header_tokens": header,
                           "sequence_tokens": expected, "processed_tokens": actual,
                           "options": len(q["criteria"]), "silent_truncation": False}
        result = {"model": CONFIG["model"], "answers": answers,
                  "usage": {"input_tokens": n_tokens, "output_tokens": 0}}
        saved = {"tag": tag, "request_sha256": digest, "request": payload, "response": result,
                 "token_audit": audits, "elapsed_seconds": time.monotonic() - started}
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False)
        tmp.replace(path)
        self.calls.append(saved)
        return result


def shared_context(history, budget):
    shared = ev.context(history)
    while len(budget.state_tokens(shared)) > CONFIG["shared_token_budget"]:
        records = [shared["task_start_excerpt"]] + shared["trajectory_end_excerpt"]
        if max(len(r["content"].encode()) for r in records) <= 100:
            raise ValueError("Shared metadata alone exceeds budget")
        for r in records:
            r["content"] = ev.clip(r["content"], max(100, len(r["content"].encode()) // 2))
    return shared


def token_chunks(history, shared, budget):
    """Every original content character is passed to recall, with original step IDs."""
    batch = []
    for h in history:
        original = h["content"]
        pending = [(original, 0)]
        delivered = []
        while pending:
            content, start = pending.pop()
            r = dict(ev.record(h), content=content, content_start=start, content_end=start + len(content))
            q = ev.questions([h["step"]])
            if not budget.fits(dict(shared, segment=[r]), q):
                if len(content) < 2:
                    raise ValueError("Cannot fit even one trajectory character")
                mid = len(content) // 2
                pending.extend([(content[mid:], start + mid), (content[:mid], start)])
                continue
            candidate = batch + [r]
            ids = sorted({x["step"] for x in candidate})
            if batch and (len(ids) > CONFIG["max_recall_options"] or
                          not budget.fits(dict(shared, segment=candidate), ev.questions(ids))):
                yield batch
                batch = []
            batch.append(r)
            delivered.append(content)
        if "".join(delivered) != original:
            raise ValueError("Recall segmentation lost or reordered text")
    if batch:
        yield batch


def fitted_evidence(history, candidates, shared, questions, budget):
    state = dict(shared, candidate_evidence=[ev.evidence(history, s) for s in candidates])
    original_bytes = len(ev.dumps(state).encode())
    rounds = 0
    while not budget.fits(state, questions):
        rounds += 1
        if rounds > 30:
            raise ValueError("Cannot fit evidence metadata")
        for item in state["candidate_evidence"]:
            records = [item["candidate"]] + item["neighbors"]
            if "preceding_handoff" in item:
                records.append(item["preceding_handoff"])
            for r in records:
                r["content"] = ev.clip(r["content"], max(80, int(len(r["content"].encode()) * .75)))
    return state, {"shrink_rounds": rounds, "initial_bytes": original_bytes,
                   "supplied_bytes": len(ev.dumps(state).encode())}


def predict(history, client):
    names = ev.roles(history)
    ids = [h["step"] for h in history]
    direct = {"history": [ev.record(h) for h in history], "known_outcome": "Failed execution"}
    qs = ev.questions(ids, names)
    adaptations = []
    if len(ids) <= CONFIG["max_recall_options"] and client.budget.fits(direct, qs):
        response = client.call(direct, qs, "direct")
        recall = expanded = candidates = ids
        method, count = "full_context", 1
    else:
        shared = shared_context(history, client.budget)
        recall, count = set(), 0
        for i, batch in enumerate(token_chunks(history, shared, client.budget)):
            candidates = sorted({h["step"] for h in batch})
            response = client.call(dict(shared, segment=batch), ev.questions(candidates), "recall_%03d" % i)
            recall.update(ev.top_steps(response["answers"]["root_step"], candidates, CONFIG["recall_k"]))
            count += 1
        recall = sorted(recall)
        candidates = set(recall)
        for s in recall:
            handoff = ev.nearest_handoff(history, s)
            if handoff is not None:
                candidates.add(handoff)
        expanded = candidates = sorted(candidates)
        level = 0
        while len(candidates) > CONFIG["final_group_size"]:
            reduced = set()
            for i in range(0, len(candidates), CONFIG["final_group_size"]):
                group = candidates[i:i + CONFIG["final_group_size"]]
                if len(group) <= CONFIG["recall_k"]:
                    reduced.update(group)
                    continue
                qs = ev.questions(group)
                state, audit = fitted_evidence(history, group, shared, qs, client.budget)
                tag = "reduce_%02d_%03d" % (level, i)
                adaptations.append(dict(audit, tag=tag))
                response = client.call(state, qs, tag)
                reduced.update(ev.top_steps(response["answers"]["root_step"], group, CONFIG["recall_k"]))
            candidates = sorted(reduced)
            level += 1
        qs = ev.questions(candidates, names)
        state, audit = fitted_evidence(history, candidates, shared, qs, client.budget)
        adaptations.append(dict(audit, tag="final"))
        response = client.call(state, qs, "final")
        method = "segmented_choice"
    a = response["answers"]
    return {"predicted_role": a["responsible_role"]["choice"],
            "predicted_step": ev.top_steps(a["root_step"], candidates, 1)[0],
            "role_confidence": a["responsible_role"].get("confidence"),
            "step_confidence": a["root_step"].get("confidence"),
            "method": method, "segments": count, "recall_candidates": recall,
            "expanded_candidates": expanded, "final_candidates": candidates,
            "evidence_adaptations": adaptations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="results/laya_full")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from rl_agent_api import RLAgent
    torch.set_num_threads(2)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("Expected an available local GPU")
    ev.CONFIG.update(CONFIG)
    rows, manifest = load_full()
    if args.smoke:
        rows = [min((r for r in rows if r["source"] == s), key=lambda r: len(ev.dumps(r["history"]).encode()))
                for s in sorted({r["source"] for r in rows})]
    out = ROOT / args.output
    config = dict(CONFIG, dataset_revision=manifest["revision"], subset="full",
                  runner_sha256=ev.sha(Path(__file__).read_bytes()),
                  upstream_source_sha256={p.name: ev.sha(p.read_bytes()) for p in
                                          (MODEL_DIR / "rl_agent_api.py", MODEL_DIR / "rl_common.py")})
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Output protocol/code changed; choose a new output")
    else:
        # All shards use identical bytes; unique temporary names avoid write races.
        tmp = out / ("config.%d.tmp" % os.getpid())
        tmp.write_text(json.dumps(config, indent=2) + "\n")
        tmp.replace(config_path)
    started = time.monotonic()
    agent = RLAgent(str(MODEL_DIR), device="cuda:0")
    runtime = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "shard": args.shard, "shards": args.shards, "pid": os.getpid(),
               "started_unix": time.time(), "parameters": sum(p.numel() for p in agent.model.parameters())}
    runtime_path = out / ("runtime_%02d.json" % args.shard)
    prior = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    attempt = dict(runtime)
    runtime["attempts"] = prior.get("attempts", []) + [attempt]
    runtime["started_unix"] = prior.get("started_unix", runtime["started_unix"])
    atomic_json(runtime_path, runtime)
    print(json.dumps(runtime), flush=True)
    selected = rows[args.shard::args.shards]
    for index, row in enumerate(selected):
        qid = row["question_ID"]
        target = out / "predictions" / (qid + ".json")
        if target.exists():
            continue
        client = LocalClient(agent, out / "calls" / qid)
        begin = time.monotonic()
        try:
            prediction = dict(predict(row["history"], client), status="ok")
        except Exception as error:
            import traceback
            atomic_json(out / "errors" / (qid + ".json"), {"error": str(error), "traceback": traceback.format_exc()})
            raise
        prediction.update(question_ID=qid, elapsed_seconds=time.monotonic() - begin,
                          local_calls=len(client.calls),
                          input_tokens=sum(c["response"]["usage"]["input_tokens"] for c in client.calls))
        atomic_json(target, prediction)
        print(json.dumps({"done": index + 1, "assigned": len(selected), "qid": qid,
                          "seconds": round(prediction["elapsed_seconds"], 2),
                          "calls": len(client.calls), "elapsed": round(time.monotonic()-started, 1)}), flush=True)
    runtime.update(finished_unix=time.time(), elapsed_seconds=time.monotonic()-started,
                   peak_cuda_memory_bytes=max(prior.get("peak_cuda_memory_bytes", 0), torch.cuda.max_memory_allocated()))
    runtime["attempts"][-1]["finished_unix"] = runtime["finished_unix"]
    atomic_json(runtime_path, runtime)


if __name__ == "__main__":
    main()
