"""Local tree of addressable evidence; previews are exact excerpts, not summaries."""
import math
import re

import evaluate as ev
from rcta_evidence import terms

SIGNALS = {
    "failure": re.compile(r"error|fail|exception|mismatch|missing|失败|错误|缺失", re.I),
    "repair": re.compile(r"fix|repair|correct|validat|修复|验证", re.I),
    "constraint": re.compile(r"must|required|should|constraint|必须|要求", re.I),
}


class Directory:
    def __init__(self, index, fanout=6):
        self.index, self.fanout = index, fanout
        self.nodes = {}
        self.query = terms(index.history[0]["content"] + " " + index.history[-1]["content"])
        self.root = self._build(0, len(index.segments))

    def _build(self, start, end):
        key = "range_%d_%d" % (start, end)
        node = {"id": key, "start_segment": start, "end_segment": end, "children": []}
        self.nodes[key] = node
        if end - start > 1:
            width = math.ceil((end - start) / self.fanout)
            node["children"] = [self._build(i, min(i + width, end)) for i in range(start, end, width)]
        return key

    def refs(self, key):
        node = self.nodes[key]
        return [r for s in self.index.segments[node["start_segment"]:node["end_segment"]] for r in s["refs"]]

    def excerpt(self, ref, budget=420):
        """A UTF-8 aligned excerpt with its own absolute byte interval and hash."""
        record = self.index.read(ref)
        text = record["content"]
        match = next((pattern.search(text) for pattern in SIGNALS.values() if pattern.search(text)), None)
        offset_chars = max(0, match.start() - 50) if match else 0
        prefix = text[:offset_chars].encode()
        raw = text[offset_chars:].encode()[:budget].decode("utf-8", errors="ignore").encode()
        start = record["start_byte"] + len(prefix)
        return {"source_ref": ref, "step": record["step"], "name": record["name"],
                "start_byte": start, "end_byte": start + len(raw), "sha256": ev.sha(raw),
                "content": raw.decode(), "excerpt_only": True}

    def preview(self, key):
        refs = self.refs(key)
        ranked = sorted(refs, key=lambda r: (
            -sum(bool(p.search(self.index.read(r)["content"])) for p in SIGNALS.values()),
            -len(self.query & self.index.tokens[r]), self.index.spans[r]["step"], r))
        # A bounded routing view, never a claim that the remaining text was read.
        examples = list(dict.fromkeys(ranked[:1] + [refs[len(refs) // 2]]))
        tags = [name for name, pattern in SIGNALS.items()
                if any(pattern.search(self.index.read(r)["content"]) for r in refs)]
        return {"id": key, "step_range": [self.index.spans[refs[0]]["step"], self.index.spans[refs[-1]]["step"]],
                "segments": self.nodes[key]["end_segment"] - self.nodes[key]["start_segment"],
                "local_keyword_tags": tags, "excerpts": [self.excerpt(r) for r in examples],
                "notice": "Keyword hints and sampled original excerpts; not an exhaustive causal assessment."}

    def audit(self):
        leaves = [n["start_segment"] for n in self.nodes.values() if not n["children"]]
        if sorted(leaves) != list(range(len(self.index.segments))):
            raise ValueError("Routing tree lost or duplicated a segment")
        return {"directory_nodes": len(self.nodes), "leaves": len(leaves),
                "all_segments_reachable": True, "model_text_coverage": "selective, not exhaustive"}
