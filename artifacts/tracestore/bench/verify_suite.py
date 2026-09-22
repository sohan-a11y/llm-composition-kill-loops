"""The suite must be correct before it can measure anything:
   every `buggy` MUST fail its test, every `fixed` MUST pass."""
import subprocess, sys, tempfile, os, textwrap
from bugs import BUGS

def run_test(src, test_src, timeout=10):
    """Execute source + test in a clean subprocess. Returns (passed, output)."""
    prog = src + "\n" + test_src + "\n" + \
           "\n".join(f"{n}()" for n in _test_names(test_src)) + "\nprint('ALLPASS')\n"
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return ("ALLPASS" in r.stdout), (r.stdout + r.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)

def _test_names(test_src):
    return [l.split("(")[0].replace("def ","").strip()
            for l in test_src.splitlines() if l.startswith("def test")]

if __name__ == "__main__":
    bad = 0
    print(f"{'id':<28}{'cat':<18}{'buggy fails?':>14}{'fixed passes?':>15}")
    print("-"*76)
    for b in BUGS:
        bf, _ = run_test(b["buggy"], b["test"])
        fp, out = run_test(b["fixed"], b["test"])
        ok_bug = (bf is False)
        ok_fix = (fp is True)
        if not (ok_bug and ok_fix): bad += 1
        print(f"{b['id']:<28}{b['cat']:<18}{('yes' if ok_bug else 'NO — BAD'):>14}"
              f"{('yes' if ok_fix else 'NO — BAD'):>15}")
        if not ok_fix: print(f"    fixed output: {out[-200:]}")
    print("-"*76)
    print(f"{len(BUGS)} bugs, {bad} malformed")
    sys.exit(1 if bad else 0)
