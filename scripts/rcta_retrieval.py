"""Local BM25 views, reciprocal-rank fusion and diversity-aware evidence packing.

RRF follows Cormack et al. (SIGIR 2009); MMR follows Carbonell & Goldstein
(SIGIR 1998) with lexical Jaccard similarity. No learned retriever is claimed.
"""
from collections import Counter
import math
import re

import evaluate as ev


def tokens(text):
    words = re.findall(r"[a-z0-9_./:-]{2,}", text.casefold())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return words


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, ref in enumerate(dict.fromkeys(ranking), 1):
            scores[ref] = scores.get(ref, 0.0) + 1 / (k + rank)
    return scores


class Retrieval:
    def __init__(self, index, rrf_k=60, diversity_weight=.7):
        self.index, self.rrf_k, self.weight = index, rrf_k, diversity_weight
        self.text = {r: index.read(r)["content"] for r in index.spans}
        self.tf = {r: Counter(tokens(t)) for r, t in self.text.items()}
        self.lengths = {r: sum(c.values()) for r, c in self.tf.items()}
        self.avg_length = max(sum(self.lengths.values()) / max(len(self.tf), 1), 1)
        df = Counter(t for c in self.tf.values() for t in c)
        self.idf = {t: math.log(1 + (len(self.tf) - n + .5) / (n + .5)) for t, n in df.items()}
        self.last_rankings = []

    def bm25(self, query, eligible):
        wanted = sorted(set(tokens(query)))
        scored = []
        for ref in eligible:
            tf = self.tf[ref]
            normalizer = 1.2 * (.25 + .75 * self.lengths[ref] / self.avg_length)
            score = math.fsum(self.idf.get(t, 0) * tf[t] * 2.2 / (tf[t] + normalizer)
                              for t in wanted if tf[t])
            if score > 0:
                scored.append((-score, self.index.spans[ref]["step"], ref))
        return [v[-1] for v in sorted(scored)]

    def rankings(self, query_texts, eligible=None):
        eligible = list(self.index.spans) if eligible is None else list(eligible)
        rankings, seen = [], set()
        for query in query_texts:
            ranking = self.bm25(query, eligible)
            key = tuple(ranking)
            if ranking and key not in seen:
                rankings.append(ranking)
                seen.add(key)
        self.last_rankings = rankings
        scores = reciprocal_rank_fusion(rankings, self.rrf_k)
        # No lexical matches is a retrieval failure, not proof of irrelevance.
        return scores or dict.fromkeys(eligible, 1.0)

    def ranked(self, queries, eligible=None):
        scores = self.rankings(queries, eligible)
        return sorted(scores, key=lambda r: (-scores[r], self.index.spans[r]["step"], r))

    def diverse(self, scores, selected=()):
        """Yield until the caller's byte budget is full; no fixed candidate top-k."""
        pending, previous = set(scores), list(selected)
        maximum = max(scores.values(), default=1) or 1
        while pending:
            def priority(ref):
                own = set(self.tf[ref])
                similarity = max((len(own & set(self.tf[r])) / max(len(own | set(self.tf[r])), 1)
                                  for r in previous), default=0)
                if any(self.index.spans[r]["step"] == self.index.spans[ref]["step"] for r in previous):
                    similarity = 1
                value = self.weight * scores[ref] / maximum - (1 - self.weight) * similarity
                return (-value, self.index.spans[ref]["step"], ref)
            ref = min(pending, key=priority)
            pending.remove(ref)
            previous.append(ref)
            yield ref

    def excerpt(self, ref, query, byte_limit=650):
        text, wanted = self.text[ref], set(tokens(query))
        # Extract an original window near a distinctive query match, rather
        # than always showing the beginning of a long source fragment.
        matches = [(self.idf.get(m.group().casefold(), 0), -m.start(), m.start())
                   for m in re.finditer(r"[a-zA-Z0-9_./:-]{2,}", text)
                   if m.group().casefold() in wanted]
        start_chars = max(0, max(matches)[2] - 90) if matches else 0
        raw = text[start_chars:].encode()[:byte_limit].decode("utf-8", errors="ignore").encode()
        span = self.index.spans[ref]
        start = span["start_byte"] + len(text[:start_chars].encode())
        return {"source_ref": ref, "step": span["step"], "name": self.index.history[span["step"]]["name"],
                "start_byte": start, "end_byte": start + len(raw), "sha256": ev.sha(raw),
                "content": raw.decode(), "excerpt_only": True}

    def packet(self, queries, byte_budget, excluded_steps=(), per_excerpt=650):
        excluded = set(excluded_steps)
        eligible = [r for r, s in self.index.spans.items() if s["step"] not in excluded]
        # A short, fully readable trajectory should not lose a decisive action
        # just because it has no query-word overlap. Route by actual view size.
        full, full_size = [], 2
        for ref in eligible:
            record = dict(self.index.read(ref), source_ref=ref, excerpt_only=False)
            added = len(ev.dumps(record).encode()) + 1
            if full_size + added > byte_budget:
                break
            full.append(record)
            full_size += added
        if len(full) == len(eligible):
            self.last_rankings = []
            return {"records": full, "ranked_spans": len(eligible), "query_count": 0,
                    "view_complete": True, "retrieval_strategy": "all_eligible_original_text",
                    "notice": "All eligible original spans fit; no lexical filtering or excerpt clipping applied."}
        scores = self.rankings(queries, eligible)
        cards, size = [], 2
        for ref in self.diverse(scores):
            card = self.excerpt(ref, " ".join(queries), per_excerpt)
            added = len(ev.dumps(card).encode()) + 1
            if size + added > byte_budget:
                break
            cards.append(card)
            size += added
        return {"records": cards, "ranked_spans": len(scores), "query_count": len(self.last_rankings),
                "retrieval_strategy": "global_rrf_mmr_excerpts",
                "view_complete": len(cards) == len(eligible),
                "notice": "Global lexical retrieval with RRF/MMR. Excerpts are leads, not proof or exhaustive coverage."}
