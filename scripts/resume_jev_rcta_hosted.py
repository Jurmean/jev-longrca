"""Transport-only recovery for response-less disconnects; inference stays frozen."""
import http.client
import json
from pathlib import Path
import sys
import time

import evaluate as ev
from evaluate_jev_rcta import write_json
from jev_rcta_client import Client, BalanceExhausted
import run_jev_rcta_hosted_full as runner


class RecoveryClient(Client):
    def call(self, state, questions, tag):
        for attempt in range(4):
            try:
                return super().call(state, questions, tag)
            except (ConnectionError, http.client.HTTPException) as error:
                if attempt == 3:
                    raise RuntimeError("Response-less connection failure after bounded retries") from None
                event = {"question_ID": self.directory.name, "tag": tag,
                         "error_type": type(error).__name__, "retry_seconds": 5 * (attempt + 1),
                         "time_unix": time.time(), "note": "Identical endpoint, model and payload; no response received."}
                with (self.run_directory / "disconnect_retries.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                print("  Response-less disconnect; bounded transport retry", flush=True)
                if self.stop_event.wait(event["retry_seconds"]):
                    raise RuntimeError("Run stopped; disconnect retry cancelled") from None


def main():
    # Parse only output here; the frozen runner owns all other arguments.
    output = "results/jev_rcta_hosted_full"
    if "--output" in sys.argv:
        output = sys.argv[sys.argv.index("--output") + 1]
    out = ev.ROOT / output
    if not (out / "config.json").exists():
        raise SystemExit("Recovery requires an existing frozen run")
    recovery = {"adapter_sha256": ev.sha(Path(__file__).read_bytes()),
                "base_client_sha256": ev.sha((ev.ROOT / "scripts/jev_rcta_client.py").read_bytes()),
                "method_sha256": ev.sha((ev.ROOT / "scripts/jev_rcta.py").read_bytes()),
                "retry_seconds": [5, 10, 15],
                "note": "Transport-only recovery; no change to model, endpoint, request content or scoring."}
    target = out / "transport_recovery.json"
    if target.exists() and json.loads(target.read_text()) != recovery:
        raise ValueError("Recovery adapter changed")
    write_json(target, recovery)
    runner.Client = RecoveryClient
    runner.main()


if __name__ == "__main__":
    main()
