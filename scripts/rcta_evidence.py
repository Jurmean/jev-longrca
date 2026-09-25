"""Lossless, byte-addressed trajectory store and deterministic evidence retrieval."""
import collections
import math
import re

import evaluate as ev

STOPWORDS = set("the a an to of for is are was be and or in on it this that with from as by not".split())
REPAIR = re.compile(r"fix|repair|correct|test|verify|validat|pass|fail|修复|验证|测试|失败", re.I)


def terms(text):
    return set(re.findall(r"[\w./:-]{3,}", text.casefold())) - STOPWORDS


class EvidenceIndex:
    def __init__(self, history, span_bytes=2200, segment_bytes=15000, segment_records=16):
        if not history or [h.get("step") for h in history] != list(range(len(history))):
            raise ValueError("Expected non-empty contiguous 0-based history")
        if span_bytes < 32 or segment_bytes < span_bytes * 2 + 800 or segment_records < 1:
            raise ValueError("Invalid evidence budgets")
        self.history = [ev.record(h) for h in history]
        if any(not isinstance(h["content"], str) for h in self.history):
            raise ValueError("History content must be text")
        self.history_sha256 = ev.sha(ev.dumps(self.history).encode())
        self.raw = {h["step"]: h["content"].encode() for h in self.history}
        self.spans, self.by_step, self.segments, self.tokens = {}, collections.defaultdict(list), [], {}
        batch, batch_bytes = [], 2
        for h in self.history:
            offset = 0
            pending = collections.deque(ev.split_text(h["content"], span_bytes))
            while pending:
                part = pending.popleft()
                raw = part.encode()
                ref = "s%d:%d:%d" % (h["step"], offset, offset + len(raw))
                span = {"id": ref, "step": h["step"], "start_byte": offset,
                        "end_byte": offset + len(raw), "sha256": ev.sha(raw)}
                rendered = dict(span, name=h["name"], role=h["role"], content=part)
                rendered_bytes = len(ev.dumps(rendered).encode())
                if rendered_bytes > min(segment_bytes - 2, span_bytes + 350):
                    if len(part) <= 1:
                        raise ValueError("Record metadata cannot fit evidence budget")
                    middle = len(part) // 2
                    pending.appendleft(part[middle:])
                    pending.appendleft(part[:middle])
                    continue
                self.spans[ref] = span
                self.by_step[h["step"]].append(ref)
                self.tokens[ref] = terms(part)
                offset += len(raw)
                # Handoffs and new task messages are natural optional boundaries.
                boundary = self.spans[ref]["start_byte"] == 0 and (
                    "(->" in h["name"] or h["role"] in ("user", "system"))
                oversized = batch_bytes + rendered_bytes + bool(batch) > segment_bytes
                if batch and (oversized or len(batch) >= segment_records or
                              boundary and batch_bytes >= segment_bytes // 2):
                    self.segments.append({"id": "seg%04d" % len(self.segments), "refs": batch})
                    batch, batch_bytes = [], 2
                batch_bytes += rendered_bytes + bool(batch)
                batch.append(ref)
        if batch:
            self.segments.append({"id": "seg%04d" % len(self.segments), "refs": batch})
        counts = collections.Counter(t for tokens in self.tokens.values() for t in tokens)
        self.idf = {t: math.log(1 + len(self.spans) / n) for t, n in counts.items()}

    def read(self, ref):
        span = self.spans[ref]
        h = self.history[span["step"]]
        raw = self.raw[span["step"]][span["start_byte"]:span["end_byte"]]
        if ev.sha(raw) != span["sha256"]:
            raise ValueError("Evidence hash mismatch")
        return dict(span, name=h["name"], role=h["role"], content=raw.decode())

    def packet(self, refs, budget):
        """Whole spans only: do not silently clip a citation's content."""
        records, size = [], 2
        unique = list(dict.fromkeys(refs))
        for ref in unique:
            item = self.read(ref)
            added = len(ev.dumps(item).encode()) + 1
            if size + added > budget:
                break
            records.append(item)
            size += added
        return {"records": records, "remaining_spans": len(unique) - len(records),
                "view_complete": len(records) == len(unique)}

    def retrieve(self, step, kind, seen=()):
        """Rank all eligible spans; a page limit is a budget, not a causal edge."""
        seen = set(seen)
        query = set().union(*(self.tokens[r] for r in self.by_step[step]))
        target = ev.normalize_role(self.history[step]["name"])
        ranked = []
        for ref, span in self.spans.items():
            s = span["step"]
            if ref in seen or s == step:
                continue
            if kind == "upstream" and s >= step or kind == "repair" and s <= step:
                continue
            h = self.history[s]
            recipient = re.search(r"\(\s*->\s*(.*?)\)", h["name"])
            addressed = bool(recipient and ev.normalize_role(recipient.group(1)) == target)
            # Stable summation also preserves exact cache replay after restarting
            # Python with a different hash seed.
            score = math.fsum(self.idf[t] for t in sorted(query & self.tokens[ref]))
            if kind == "upstream" and addressed:
                score += 12
            if kind in ("repair", "challenge") and REPAIR.search(self.read(ref)["content"]):
                score += 3
            # Temporal proximity is only a retrieval fallback; the model must
            # establish a relevant relation from the actual text in another call.
            ranked.append((-score, abs(s - step), span["start_byte"], ref))
        return [r[-1] for r in sorted(ranked)]

    def audit(self):
        for h in self.history:
            if "".join(self.read(r)["content"] for r in self.by_step[h["step"]]) != h["content"]:
                raise ValueError("Evidence index lost source text")
        refs = [r for segment in self.segments for r in segment["refs"]]
        if len(refs) != len(set(refs)) or set(refs) != set(self.spans):
            raise ValueError("Segment coverage is not exactly once")
        return {"steps": len(self.history), "segments": len(self.segments),
                "spans": len(self.spans), "primary_text_coverage": "lossless",
                "history_sha256": self.history_sha256}
