def running_median(nums):
    """Intentionally buggy: uses the wrong index for even-length windows."""
    out = []
    window = []
    for n in nums:
        window.append(n)
        window.sort()
        mid = len(window) // 2
        out.append(window[mid])          # BUG: for even lengths should average mid-1, mid
    return out

def normalize(rows):
    total = sum(r["v"] for r in rows)
    scaled = []
    for r in rows:
        scaled.append({"k": r["k"], "v": r["v"] / total})   # BUG: ZeroDivisionError if empty
    return scaled
