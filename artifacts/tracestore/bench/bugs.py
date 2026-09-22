"""
Bug suite for the fix-rate benchmark.

Two categories, deliberately:

  trace_revealing  the failure shows up as a wrong intermediate VALUE that an
                   execution trace makes visible (wrong accumulator, off-by-one
                   index, state not reset, wrong variable used).

  trace_neutral    the failure is visible from the source alone, or lives on a
                   path the failing call never executes. A trace should help
                   little or not at all.

Including the second category is the point. A benchmark that only contains bugs
traces are good at would measure the suite, not the tool. If traces help on both
categories equally, something is wrong with the experiment; if they help on
neither, the tool does not work.

Each entry ships the buggy source, a test, and a concrete call that triggers the
failure (used to capture the trace). `fixed` is never shown to the model -- it
exists so the suite itself can be verified correct.
"""

BUGS = [
# ---------------------------------------------------------------- revealing
dict(
  id="acc_not_reset", cat="trace_revealing",
  buggy='''
def group_sums(rows):
    """Sum values per key, in order of first appearance."""
    out = []
    total = 0
    for key, vals in rows:
        for v in vals:
            total += v
        out.append((key, total))
    return out
''',
  fixed='''
def group_sums(rows):
    out = []
    for key, vals in rows:
        total = 0
        for v in vals:
            total += v
        out.append((key, total))
    return out
''',
  test='''
def test_group_sums():
    assert group_sums([("a",[1,2]),("b",[10])]) == [("a",3),("b",10)]
    assert group_sums([("x",[5]),("y",[5]),("z",[5])]) == [("x",5),("y",5),("z",5)]
''',
  call=("group_sums", [[("a",[1,2]),("b",[10])]], {})),

dict(
  id="off_by_one_window", cat="trace_revealing",
  buggy='''
def max_window(nums, k):
    """Maximum sum of any k consecutive elements."""
    best = None
    for i in range(len(nums) - k):
        s = sum(nums[i:i+k])
        if best is None or s > best:
            best = s
    return best
''',
  fixed='''
def max_window(nums, k):
    best = None
    for i in range(len(nums) - k + 1):
        s = sum(nums[i:i+k])
        if best is None or s > best:
            best = s
    return best
''',
  test='''
def test_max_window():
    assert max_window([1,2,3,4], 2) == 7
    assert max_window([5,1,1], 3) == 7
''',
  call=("max_window", [[1,2,3,4], 2], {})),

dict(
  id="wrong_var_used", cat="trace_revealing",
  buggy='''
def scale_to_max(nums):
    """Divide every element by the largest element."""
    if not nums:
        return []
    top = max(nums)
    first = nums[0]
    return [n / first for n in nums]
''',
  fixed='''
def scale_to_max(nums):
    if not nums:
        return []
    top = max(nums)
    return [n / top for n in nums]
''',
  test='''
def test_scale_to_max():
    assert scale_to_max([2,4]) == [0.5, 1.0]
    assert scale_to_max([10,5]) == [1.0, 0.5]
''',
  call=("scale_to_max", [[2,4]], {})),

dict(
  id="shared_mutable", cat="trace_revealing",
  buggy='''
def bucket(items, n):
    """Distribute items round-robin into n buckets."""
    buckets = [[]] * n
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets
''',
  fixed='''
def bucket(items, n):
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets
''',
  test='''
def test_bucket():
    assert bucket([1,2,3,4], 2) == [[1,3],[2,4]]
''',
  call=("bucket", [[1,2,3,4], 2], {})),

dict(
  id="running_median_even", cat="trace_revealing",
  buggy='''
def running_median(nums):
    """Median of all elements seen so far, at each step."""
    out = []
    window = []
    for n in nums:
        window.append(n)
        window.sort()
        mid = len(window) // 2
        out.append(float(window[mid]))
    return out
''',
  fixed='''
def running_median(nums):
    out = []
    window = []
    for n in nums:
        window.append(n)
        window.sort()
        mid = len(window) // 2
        if len(window) % 2 == 0:
            out.append((window[mid-1] + window[mid]) / 2)
        else:
            out.append(float(window[mid]))
    return out
''',
  test='''
def test_running_median():
    assert running_median([1,3]) == [1.0, 2.0]
    assert running_median([5,1,9]) == [5.0, 3.0, 5.0]
''',
  call=("running_median", [[1,3]], {})),

dict(
  id="early_return_in_loop", cat="trace_revealing",
  buggy='''
def all_positive(nums):
    """True only if every element is > 0."""
    for n in nums:
        if n > 0:
            return True
        else:
            return False
    return True
''',
  fixed='''
def all_positive(nums):
    for n in nums:
        if n <= 0:
            return False
    return True
''',
  test='''
def test_all_positive():
    assert all_positive([1,2,3]) is True
    assert all_positive([1,-1]) is False
    assert all_positive([]) is True
''',
  call=("all_positive", [[1,-1]], {})),

dict(
  id="stale_index", cat="trace_revealing",
  buggy='''
def pair_with_next(items):
    """Pair each item with the one after it."""
    out = []
    for i in range(len(items)):
        if i + 1 < len(items):
            out.append((items[i], items[i]))
    return out
''',
  fixed='''
def pair_with_next(items):
    out = []
    for i in range(len(items)):
        if i + 1 < len(items):
            out.append((items[i], items[i+1]))
    return out
''',
  test='''
def test_pair_with_next():
    assert pair_with_next([1,2,3]) == [(1,2),(2,3)]
''',
  call=("pair_with_next", [[1,2,3]], {})),

dict(
  id="count_overwrite", cat="trace_revealing",
  buggy='''
def char_counts(s):
    """Count occurrences of each character."""
    counts = {}
    for ch in s:
        counts[ch] = 1
    return counts
''',
  fixed='''
def char_counts(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
''',
  test='''
def test_char_counts():
    assert char_counts("aab") == {"a":2, "b":1}
''',
  call=("char_counts", ["aab"], {})),

dict(
  id="accumulate_wrong_op", cat="trace_revealing",
  buggy='''
def compound(rate, years):
    """Growth factor after `years` of compounding at `rate`."""
    factor = 1.0
    for _ in range(years):
        factor += rate
    return factor
''',
  fixed='''
def compound(rate, years):
    factor = 1.0
    for _ in range(years):
        factor *= (1 + rate)
    return factor
''',
  test='''
def test_compound():
    assert abs(compound(0.1, 2) - 1.21) < 1e-9
    assert abs(compound(0.0, 5) - 1.0) < 1e-9
''',
  call=("compound", [0.1, 2], {})),

dict(
  id="reversed_accumulation", cat="trace_revealing",
  buggy='''
def prefix_max(nums):
    """Largest value seen up to and including each position."""
    out = []
    best = 0
    for n in nums:
        best = n
        out.append(best)
    return out
''',
  fixed='''
def prefix_max(nums):
    out = []
    best = None
    for n in nums:
        best = n if best is None else max(best, n)
        out.append(best)
    return out
''',
  test='''
def test_prefix_max():
    assert prefix_max([1,5,2,9]) == [1,5,5,9]
    assert prefix_max([3,1]) == [3,3]
''',
  call=("prefix_max", [[1,5,2,9]], {})),

# ------------------------------------------------------------------ neutral
dict(
  id="wrong_constant", cat="trace_neutral",
  buggy='''
def celsius_to_f(c):
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32.5
''',
  fixed='''
def celsius_to_f(c):
    return c * 9 / 5 + 32
''',
  test='''
def test_celsius_to_f():
    assert celsius_to_f(0) == 32
    assert celsius_to_f(100) == 212
''',
  call=("celsius_to_f", [0], {})),

dict(
  id="inverted_comparison", cat="trace_neutral",
  buggy='''
def is_adult(age):
    """True if age is 18 or more."""
    return age > 18
''',
  fixed='''
def is_adult(age):
    return age >= 18
''',
  test='''
def test_is_adult():
    assert is_adult(18) is True
    assert is_adult(17) is False
''',
  call=("is_adult", [18], {})),

dict(
  id="missing_guard_unexecuted", cat="trace_neutral",
  buggy='''
def safe_div(a, b):
    """Return a/b, or None when b is zero."""
    return a / b
''',
  fixed='''
def safe_div(a, b):
    if b == 0:
        return None
    return a / b
''',
  test='''
def test_safe_div():
    assert safe_div(6, 3) == 2
    assert safe_div(1, 0) is None
''',
  call=("safe_div", [6, 3], {})),

dict(
  id="wrong_string_method", cat="trace_neutral",
  buggy='''
def slugify(text):
    """Lowercase, spaces to hyphens."""
    return text.upper().replace(" ", "-")
''',
  fixed='''
def slugify(text):
    return text.lower().replace(" ", "-")
''',
  test='''
def test_slugify():
    assert slugify("Hello World") == "hello-world"
''',
  call=("slugify", ["Hello World"], {})),

dict(
  id="swapped_args", cat="trace_neutral",
  buggy='''
def clamp(value, low, high):
    """Constrain value to [low, high]."""
    return max(high, min(low, value))
''',
  fixed='''
def clamp(value, low, high):
    return max(low, min(high, value))
''',
  test='''
def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
''',
  call=("clamp", [5, 0, 10], {})),

dict(
  id="wrong_default", cat="trace_neutral",
  buggy='''
def join_words(words, sep=" "):
    """Join words with sep; default is a comma-space."""
    return sep.join(words)
''',
  fixed='''
def join_words(words, sep=", "):
    return sep.join(words)
''',
  test='''
def test_join_words():
    assert join_words(["a","b"]) == "a, b"
    assert join_words(["a","b"], "-") == "a-b"
''',
  call=("join_words", [["a","b"]], {})),

dict(
  id="off_by_one_range_simple", cat="trace_neutral",
  buggy='''
def first_n_squares(n):
    """Squares of 1..n."""
    return [i*i for i in range(1, n)]
''',
  fixed='''
def first_n_squares(n):
    return [i*i for i in range(1, n+1)]
''',
  test='''
def test_first_n_squares():
    assert first_n_squares(3) == [1,4,9]
''',
  call=("first_n_squares", [3], {})),

dict(
  id="integer_division", cat="trace_neutral",
  buggy='''
def average(nums):
    """Arithmetic mean."""
    return sum(nums) // len(nums)
''',
  fixed='''
def average(nums):
    return sum(nums) / len(nums)
''',
  test='''
def test_average():
    assert average([1,2]) == 1.5
''',
  call=("average", [[1,2]], {})),
]


def by_category():
    out = {}
    for b in BUGS:
        out.setdefault(b["cat"], []).append(b)
    return out
