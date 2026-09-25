"""Synthetic responses to exercise v3 control flow, not model predictions."""
from rcta_demo import HISTORY, ScriptedClient as Base


class ScriptedClient(Base):
    def call(self, state, questions, tag):
        # Base records the same response object; all substitutions remain in the
        # fixture trace for deterministic replay tests.
        result = super().call(state, questions, tag)
        for key, q in questions.items():
            opts = q["criteria"]
            desired = {"origin": "local", "outcome": "persists", "next_need": "upstream"}.get(key)
            if key in ("symptom", "responsibility_citation"):
                step = 3 if key == "symptom" else 1
                desired = next((r for r in opts if r.startswith("s%d:" % step)), "unknown")
            if desired in opts:
                result["answers"][key] = {"choice": desired, "probabilities": {
                    k: .98 if k == desired else .02 / (len(opts) - 1) for k in opts}}
        return result
