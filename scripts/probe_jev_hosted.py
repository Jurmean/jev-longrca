"""Minimal authenticated compatibility check for the user-selected hosted API."""
import json
import os
from pathlib import Path

import evaluate as ev
from jev_rcta_client import Client

ENDPOINT = "https://jevtypesafeai.com/api/v1/decide"


def read_key():
    key = os.environ.get("JEV_HOSTED_API_KEY")
    if not key:
        for line in (ev.ROOT / ".env.jev-hosted").read_text().splitlines():
            if line.startswith("JEV_HOSTED_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise ValueError("Missing private JEV_HOSTED_API_KEY")
    return key


def probe(out):
    out.mkdir(parents=True, exist_ok=True)
    client = Client(read_key(), out / "calls", out, endpoint=ENDPOINT)
    questions = {
        "topic": {"type": "choice", "instructions": "Classify the customer's request.",
                  "criteria": {"billing": "Payment or refund", "technical": "Software bug"}},
        "language": {"type": "choice", "instructions": "Identify the language of the customer message.",
                     "criteria": {"English": "English", "Spanish": "Spanish"}},
    }
    result = client.call("Customer: I was charged twice. Please refund the duplicate payment.", questions, "compatibility")
    valid = result["answers"].get("topic", {}).get("choice") == "billing" and \
        result["answers"].get("language", {}).get("choice") == "English"
    report = {"passed": valid, "endpoint": ENDPOINT, "model": result["model"],
              "answers": result["answers"], "usage": result.get("usage", {})}
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not valid:
        raise RuntimeError("Control questions failed; do not launch benchmark")
    return report


if __name__ == "__main__":
    probe(ev.ROOT / "results/jev_rcta_hosted_probe")
