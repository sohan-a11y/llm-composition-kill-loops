import sys,time,statistics
sys.path.insert(0,'/home/claude/tracestore')
from tracestore.tracer import Tracer
sys.path.insert(0,'/home/claude/tracestore/demo')
import work
TARGET=work.__file__
TEXT=("the quick brown fox jumps over the lazy dog. "*40+
      "pack my box with five dozen liquor jugs! "*40)
def t(fn,n=5):
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); fn(); ts.append(time.perf_counter()-a)
    return min(ts)
base=t(lambda: work.run(TEXT,5,30))
print(f"untraced: {base*1000:.2f} ms\n")
print(f"{'per_line_cap':<16}{'ms':>9}{'overhead':>10}{'events':>9}")
print("-"*44)
for cap in [1,3,10,None]:
    h={}
    def r():
        c = 10**9 if cap is None else cap
        with Tracer(TARGET,per_line_cap=c) as tr: work.run(TEXT,5,30)
        h['t']=tr
    ms=t(r); s=h['t'].stats()
    lbl=str(cap) if cap else "uncapped"
    print(f"{lbl:<16}{ms*1000:>9.2f}{ms/base:>9.1f}x{s['events']:>9}")
