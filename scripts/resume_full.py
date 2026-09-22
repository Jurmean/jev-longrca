"""Transport-only recovery for intermittent provider failures; inference stays frozen."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import evaluate as ev
import evaluate_full

BASE_CLIENT = ev.Client
LOCK = threading.Lock()
LOG = ev.ROOT / "results/full/transport_retries.jsonl"
STOP = threading.Event()
ORIGINAL_URLOPEN = urllib.request.urlopen


class BillingStop(BaseException):
    """Escapes ordinary per-case error handling so the whole run stops."""


def insufficient_balance(status, body):
    text = body.casefold()
    return status == 402 or any(marker in text for marker in (
        "insufficient_balance", "insufficient balance", "insufficient credit", "insufficient_credit",
        "out of credits", "not enough credits", "credits exhausted", "balance exhausted",
        "exhausted your credits", "balance is zero", "insufficient_quota", "billing_hard_limit",
        "余额不足", "额度用尽"))


def guarded_urlopen(*args, **kwargs):
    if STOP.is_set():
        raise BillingStop("Balance stop already triggered")
    try:
        return ORIGINAL_URLOPEN(*args, **kwargs)
    except urllib.error.HTTPError as error:
        raw = error.read()
        if insufficient_balance(error.code, raw.decode("utf-8", "replace")):
            STOP.set()
            with LOCK:
                state = {"status": "paused_insufficient_balance", "http_status": error.code,
                         "utc": datetime.now(timezone.utc).isoformat(),
                         "completed_n": len(list((ev.ROOT / "results/full/predictions").glob("*.json"))),
                         "note": "Stopped new requests per user instruction. Await a new API configuration; preserve all checkpoints."}
                (ev.ROOT / "results/full/balance_stop.json").write_text(json.dumps(state, indent=2) + "\n")
                print("BALANCE EXHAUSTED: stopped new requests; checkpoints preserved.", flush=True)
            error.close()
            raise BillingStop("Insufficient balance; awaiting new API configuration") from None
        # Preserve the full error body for the unchanged client/retry handling.
        restored = urllib.error.HTTPError(error.url, error.code, error.msg, error.headers, io.BytesIO(raw))
        error.close()
        raise restored from None


class RecoveryClient(BASE_CLIENT):
    def call(self, state, q, tag):
        for attempt in range(4):
            if STOP.is_set():
                raise BillingStop("Insufficient balance; no more requests")
            try:
                return super().call(state, q, tag)
            except RuntimeError as error:
                message = str(error).replace(self.key, "[REDACTED]")
                transient = message.startswith(("HTTP 403:", "HTTP 500:", "HTTP 502:", "HTTP 503:",
                                                 "HTTP 504:", "HTTP 529:", "API connection failed"))
                # The same fixed-version request was observed succeeding immediately
                # after this provider routing error; do not retry other HTTP 400s.
                transient = transient or (message.startswith("HTTP 400:") and
                                           "Unknown model: " + ev.MODEL in message)
                if not transient or attempt == 3:
                    raise
                delay = (15, 30, 45)[attempt]
                event = {"utc": datetime.now(timezone.utc).isoformat(), "question_ID": self.directory.name,
                         "tag": tag, "error": message, "retry_in_seconds": delay}
                with LOCK:
                    with LOG.open("a") as handle:
                        handle.write(json.dumps(event) + "\n")
                    print("Transport recovery: %s %s; retry in %ds" % (self.directory.name, tag, delay), flush=True)
                if STOP.wait(delay):
                    raise BillingStop("Insufficient balance; cancelled retry")


if __name__ == "__main__":
    recovery = {"adapter_sha256": ev.sha(Path(__file__).read_bytes()),
                "inference_sha256": ev.sha((ev.ROOT / "scripts/evaluate.py").read_bytes()),
                "retry_delays_seconds": [15, 30, 45],
                "note": "Same endpoint, key, model, payloads and inference protocol; transport retries only."}
    (ev.ROOT / "results/full/transport_recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    ev.Client = RecoveryClient
    urllib.request.urlopen = guarded_urlopen
    try:
        evaluate_full.main()
    except BillingStop:
        raise SystemExit("Full run paused due to insufficient balance. Saved results remain available.")
