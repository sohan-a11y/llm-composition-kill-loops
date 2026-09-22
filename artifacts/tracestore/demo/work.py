"""A realistic mixed workload: loops, branches, string work, dict/list building."""
def tokenize(text):
    toks, cur = [], ""
    for ch in text:
        if ch.isalnum():
            cur += ch
        else:
            if cur: toks.append(cur); cur = ""
            if ch.strip(): toks.append(ch)
    if cur: toks.append(cur)
    return toks

def word_counts(toks):
    counts = {}
    for t in toks:
        if t.isalpha():
            counts[t] = counts.get(t, 0) + 1
    return counts

def top_k(counts, k):
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:k]

def run(text, k=5, reps=1):
    out = None
    for _ in range(reps):
        toks = tokenize(text)
        counts = word_counts(toks)
        out = top_k(counts, k)
    return out
