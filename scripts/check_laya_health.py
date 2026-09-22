"""Post-run diagnostic controls, independent of benchmark labels; no tuning."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/laya"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from rl_agent_api import RLAgent


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    agent = RLAgent(str(ROOT / "models/laya"), device="cuda:0")
    native = dict(agent.cfg)
    cases = [
        ("billing", "My credit card was charged twice for one purchase. Please refund the duplicate payment.", "billing"),
        ("account", "I forgot my password and cannot sign in to my account. Please help me reset my password.", "account"),
        ("shipping", "My package has not arrived. The tracking page says it is still in transit. Where is my delivery?", "shipping")]
    criteria = {"billing": "Payment charges and refunds", "account": "Login and password problems",
                "shipping": "Delivery and package tracking"}
    rows = []
    for name, state, expected in cases:
        for rotation in range(3):
            keys = list(criteria)
            keys = keys[rotation:] + keys[:rotation]
            q = {"route": {"type": "choice", "instructions": "Which customer support category matches this message?",
                           "criteria": {key: criteria[key] for key in keys}}}
            for max_len in (512, 8192):
                agent.cfg.update(max_len=max_len, head_max_len=192)
                start = time.monotonic()
                result = agent.system_one(state, q)
                rows.append({"case": name, "option_order": keys, "max_len": max_len,
                             "state": state, "questions": q, "expected": expected,
                             "correct": result["answers"]["route"]["choice"] == expected,
                             "response": result, "seconds": time.monotonic()-start})
    result = {"purpose": "Short English diagnostic controls, not benchmark results; no parameter selection from these checks.",
              "native_max_len": native["max_len"], "native_head_max_len": native["head_max_len"],
              "tests": rows, "correct": sum(r["correct"] for r in rows), "n": len(rows)}
    (ROOT / "reports/laya_sanity_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"correct": result["correct"], "n": len(rows),
                      "choices": [{"case": r["case"], "order": r["option_order"], "max_len": r["max_len"],
                                   "choice": r["response"]["answers"]["route"]["choice"]} for r in rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
