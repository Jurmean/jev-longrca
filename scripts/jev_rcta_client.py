"""Checkpointed transport with immediate, persistent quota stop."""
import json
import time
import threading
import urllib.error
import urllib.request

import evaluate as ev


class BalanceExhausted(RuntimeError):
    pass


def insufficient_balance(status, body):
    return status == 402 or any(marker in body.casefold() for marker in (
        "insufficient_balance", "insufficient balance", "insufficient credit", "insufficient_credit",
        "out of credits", "not enough credits", "credits exhausted", "balance exhausted",
        "exhausted your credits", "balance is zero", "insufficient_quota", "billing_hard_limit",
        "余额不足", "额度用尽"))


class Client(ev.Client):
    def __init__(self, key, directory, run_directory, endpoint=None, stop_event=None):
        super().__init__(key, directory)
        self.run_directory = run_directory
        self.stop_path = run_directory / "balance_stop.json"
        self.endpoint = endpoint or ev.ENDPOINT
        if self.endpoint not in (ev.ENDPOINT, "https://jevtypesafeai.com/api/v1/decide"):
            raise ValueError("Unrecognized API endpoint")
        self.stop_event = stop_event or threading.Event()

    def call(self, state, q, tag):
        if self.stop_path.exists() or self.stop_event.is_set():
            raise BalanceExhausted("Quota stop persists; explicitly resume only after replenishing credits")
        payload = {"model": ev.MODEL, "state": state, "questions": q}
        raw = ev.dumps(payload).encode()
        digest = ev.sha(raw)
        path = self.directory / (tag + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_sha256"] != digest or ev.sha(ev.dumps(saved["request"]).encode()) != digest:
                raise ValueError("Cached request does not match protocol")
            if saved["response"].get("model") != ev.MODEL:
                raise ValueError("Cached model mismatch")
            if saved.get("endpoint", ev.ENDPOINT) != self.endpoint:
                raise ValueError("Cached endpoint mismatch")
            self.calls.append(saved)
            return saved["response"]
        request = urllib.request.Request(self.endpoint, data=raw, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        started, errors = time.monotonic(), []
        for attempt in range(4):
            if self.stop_event.is_set() or self.stop_path.exists():
                raise BalanceExhausted("Run stopped; no further requests")
            status = 200
            try:
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        body = response.read().decode("utf-8", "replace")
                except urllib.error.HTTPError as error:
                    status = error.code
                    body = error.read().decode("utf-8", "replace")
                    error.close()
                if insufficient_balance(status, body):
                    self.stop_event.set()
                    self.stop_path.write_text(json.dumps({"status": "paused_insufficient_balance",
                        "http_status": status, "question_ID": self.directory.name, "tag": tag,
                        "time_unix": time.time(), "note": "No further requests until explicit resume."}, indent=2) + "\n")
                    raise BalanceExhausted("Insufficient API balance; stopped immediately")
                if status != 200:
                    safe = body.replace(self.key, "[REDACTED]")[:1500]
                    transient = status in (403, 429, 500, 502, 503, 504, 529) or (
                        status == 400 and "Unknown model: " + ev.MODEL in body)
                    if not transient or attempt == 3:
                        raise RuntimeError("HTTP %d: %s" % (status, safe))
                    errors.append({"status": status, "message": safe})
                else:
                    result = json.loads(body)
                    if result.get("model") != ev.MODEL or not isinstance(result.get("answers"), dict):
                        raise ValueError("Unexpected model or missing API answers")
                    saved = {"tag": tag, "endpoint": self.endpoint, "request_sha256": digest, "request": payload,
                             "response": result, "elapsed_seconds": time.monotonic() - started,
                             "attempts": attempt + 1, "retry_errors": errors}
                    temp = path.with_suffix(".tmp")
                    temp.write_text(ev.dumps(saved) + "\n")
                    temp.replace(path)
                    self.calls.append(saved)
                    print("  %s: %s" % (self.directory.name, tag), flush=True)
                    return result
            except (urllib.error.URLError, TimeoutError) as error:
                errors.append({"type": type(error).__name__})
                if attempt == 3:
                    raise RuntimeError("API connection failed after retries") from None
            # Quota errors escape above before this bounded transient retry.
            event = {"question_ID": self.directory.name, "tag": tag, "attempt": attempt + 1,
                     "error": errors[-1], "retry_seconds": 5 * (attempt + 1)}
            with (self.run_directory / "transport_retries.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print("  Temporary API error; retrying in %d seconds" % event["retry_seconds"], flush=True)
            if self.stop_event.wait(event["retry_seconds"]):
                raise BalanceExhausted("Run stopped; retry cancelled")
