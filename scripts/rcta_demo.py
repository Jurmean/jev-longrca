"""Scripted integration fixture. This is NOT a model or an accuracy benchmark."""
import evaluate as ev

HISTORY = [
    {"step": 0, "name": "User", "role": "user",
     "content": "Export orders.csv with order_id AND date columns. Preserve ISO dates."},
    {"step": 1, "name": "Planner (-> Worker)", "role": "assistant",
     "content": "Write orders.csv with order_id only. Drop the required date column."},
    {"step": 2, "name": "Worker", "role": "assistant",
     "content": "Followed Planner: exported orders.csv without the date column."},
    {"step": 3, "name": "Reviewer", "role": "tool",
     "content": "Validation FAIL: orders.csv is missing date. The error persists; no repaired file was submitted."},
    {"step": 4, "name": "Worker", "role": "assistant",
     "content": "Final artifact is still orders.csv without date. Task failed."},
]


class ScriptedClient:
    def __init__(self):
        self.calls = []

    def call(self, state, questions, tag):
        phase = tag.split("_", 1)[1]
        target = state.get("target_step")
        answers = {}
        for key, q in questions.items():
            options = q["criteria"]
            value = {"event": "execution", "phenomenon": "constraint", "relevance": "cause",
                     "need": "upstream" if target == 2 else "repair", "support": "supported",
                     "root_step": "1"}.get(key)
            if key == "assessment":
                value = "propagates" if target == 2 else "introduces" if target == 1 else "refutes"
            if key == "relation":
                value = {"upstream": "upstream" if target == 2 else "local", "repair": "persists",
                         "challenge": "supports" if target == 1 else "refutes"}[phase]
            if key in ("citation", "location"):
                desired = 2 if key == "location" else target
                if phase == "upstream":
                    desired = 1 if target == 2 else 0
                elif phase == "repair":
                    desired = 3
                elif phase == "verify":
                    desired = 1
                value = next((k for k in options if k.startswith("s%d:" % desired)), "unknown")
            if key == "responsible_role":
                value = next((k for k, text in options.items() if text == "Planner"), "unknown")
            if value not in options:
                value = "unknown" if "unknown" in options else next(iter(options))
            n = len(options)
            answers[key] = {"choice": value, "confidence": 0.95,
                            "probabilities": {k: 0.98 if k == value else 0.02 / (n - 1) for k in options}}
        response = {"model": ev.MODEL, "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0},
                    "synthetic": True}
        payload = {"model": ev.MODEL, "state": state, "questions": questions}
        self.calls.append({"tag": tag, "request": payload, "request_sha256": ev.sha(ev.dumps(payload).encode()),
                           "response": response, "synthetic": True})
        return response
