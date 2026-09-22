"""End-to-end MCP protocol test: real handshake, real tool calls."""
import io, json, sys, os
sys.path.insert(0,'/home/claude/tracestore'); sys.path.insert(0,'/home/claude/tracestore/demo')
from tracestore.server import Server

DB='/tmp/test_traces.db'
if os.path.exists(DB): os.remove(DB)
srv=Server(DB)

def rpc(method, params=None, mid=1):
    msg={"jsonrpc":"2.0","id":mid,"method":method}
    if params is not None: msg["params"]=params
    return srv.handle(msg)

ok=lambda c,m: print(("  PASS " if c else "  FAIL ")+m)

print("1. initialize")
r=rpc("initialize",{"protocolVersion":"2024-11-05","capabilities":{}})
ok(r["result"]["protocolVersion"]=="2024-11-05","protocolVersion echoed")
ok(r["result"]["serverInfo"]["name"]=="tracestore","serverInfo present")
ok("tools" in r["result"]["capabilities"],"declares tools capability")

print("2. initialized notification (must return None)")
ok(srv.handle({"jsonrpc":"2.0","method":"notifications/initialized"}) is None,"no reply to notification")

print("3. tools/list")
r=rpc("tools/list",mid=2); tools=r["result"]["tools"]
ok(len(tools)==4,f"4 tools exposed (got {len(tools)})")
for t in tools:
    ok("name" in t and "description" in t and "inputSchema" in t, f"schema ok: {t['name']}")

print("4. tools/call trace_failure (captures a live trace)")
r=rpc("tools/call",{"name":"trace_failure","arguments":{
    "target":"buggy:running_median","args_json":"[[5,1,9,3]]"}},mid=3)
txt=r["result"]["content"][0]["text"]
ok(r["result"]["isError"]==False,"not an error")
ok("out=[5, 5, 5, 5]" in txt,"trace shows the actual buggy output")

print("5. tools/call trace_failure on code that RAISES")
r=rpc("tools/call",{"name":"trace_failure","arguments":{
    "target":"buggy:normalize","args_json":'[[{"k":"a","v":0}]]'}},mid=4)
txt=r["result"]["content"][0]["text"]
ok("ZeroDivisionError" in txt,"exception captured and surfaced")

print("6. tools/call get_trace (reads from store)")
r=rpc("tools/call",{"name":"get_trace","arguments":{"function":"running_median"}},mid=5)
ok("running_median" in r["result"]["content"][0]["text"],"stored trace retrieved")

print("7. tools/call search_traces")
r=rpc("tools/call",{"name":"search_traces","arguments":{"symbol":"window"}},mid=6)
ok("window" in r["result"]["content"][0]["text"],"search finds by variable name")

print("8. tools/call trace_summary")
r=rpc("tools/call",{"name":"trace_summary","arguments":{}},mid=7)
ok("traces=" in r["result"]["content"][0]["text"],"summary returned")

print("9. error handling: bad tool name")
r=rpc("tools/call",{"name":"nope","arguments":{}},mid=8)
ok(r["result"]["isError"]==True,"unknown tool -> isError true, not a crash")

print("10. error handling: unknown method")
r=rpc("frobnicate",mid=9)
ok("error" in r and r["error"]["code"]==-32601,"JSON-RPC -32601 method not found")

print("11. full stdio round-trip")
inp=io.StringIO("\n".join([
  json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
  json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"}),
  json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/list"})])+"\n")
out=io.StringIO(); Server(DB).serve(inp,out)
lines=[l for l in out.getvalue().splitlines() if l.strip()]
ok(len(lines)==2,f"2 replies for 3 messages (notification silent) -- got {len(lines)}")
ok(all(json.loads(l).get("jsonrpc")=="2.0" for l in lines),"all replies valid JSON-RPC")
