"""Offline checks for consistency.py. No model is called: a scripted judge stands in, so this tests the plumbing
(candidate pairs, quotes attached by code, gating, status count, summary input), not the model's judgment.

    python3 evals/test_conflicts.py
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import consistency as C
import summarize as S


def teams(text):
    """Teams transcript text -> spans [{id, author, ts, text}] without the full ingest pipeline."""
    spans = []
    for blk in re.split(r"\n\s*\n", text):
        L = [l for l in blk.strip().split("\n") if l.strip()]
        if len(L) >= 5 and re.search(r"\d+:\d\d$", L[2]):
            m = re.search(r"(\d+):(\d\d)$", L[2]); secs = int(m.group(1)) * 60 + int(m.group(2))
            i = len(spans)
            spans.append(dict(id=f"b{i + 1}.s1", burst_id=f"b{i + 1}", author=L[1], text=" ".join(L[4:]), source_message_ids=[f"m{i + 1}"],
                              ts=f"2000-01-01T00:{secs // 60:02d}:{secs % 60:02d}"))
    return spans


def build(source, constraints, decision_span_text, chosen="the plan"):
    spans = teams((HERE / "sources" / source).read_text())
    by = lambda frag: next(s for s in spans if frag in s["text"])
    els = [dict(id="opt1", kind="option", title=chosen, status="chosen", spans=[], created_span=by(decision_span_text)["id"], author=by(decision_span_text)["author"], aliases=[], attrs={}, question_id=None)]
    for k, frag in enumerate(constraints):
        els.append(dict(id=f"con{k + 1}", kind="constraint", title=frag, status="active", created_span=by(frag)["id"], spans=[], attrs={}, author=by(frag)["author"], aliases=[], question_id=None))
    dec = by(decision_span_text)
    d = dict(participants=sorted({s["author"] for s in spans}), span_msgs={}, elements=els, warnings=[],
             events=[dict(kind="decide", span_id=dec["id"], participant=dec["author"], element_id="opt1",
                          payload=dict(chosen=["opt1"], conditional=False, conditions=""))],
             _enterprise=dict(source_type="transcript", subject="", speakers=sorted({s["author"] for s in spans}), observers=[],
                              msg_meta=[dict(id=s["source_message_ids"][0], author=s["author"], restricted=False, audience=[]) for s in spans],
                              spans=spans, classified=[]))
    return d


class Judge:
    """Scripted stand-in for the model. rules: {text fragment in A: (verdict, resolved_later)}."""
    def __init__(self, rules): self.rules, self.seen = rules, []
    def structured(self, *, system, user, tool_name, schema, max_tokens=0):
        res = []
        for blk in user.split("\n\nPAIR ")[0:]:
            m = re.match(r"(?:PAIR )?(\d+) \(", blk)
            if not m: continue
            a = re.search(r"A \[[^\]]*\]: (.*)", blk).group(1); self.seen.append(a)
            v = next(((v, r) for f, (v, r) in self.rules.items() if f in a), ("compatible", False))
            res.append(dict(pair=int(m.group(1)), verdict=v[0], resolved_later=v[1], why="X does not fit Y." if v[0] != "compatible" else "", resolve="Which holds?"))
        return dict(results=res)


fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: fails.append(name)

# ent-12: Nina's constraint against the announced decision
d = build("ent-12-meeting-offsite-ignored-point.txt", ["finance close runs from the 27th through November 3rd"], "let's lock it: Denver, October 29 to 31", "Denver, Oct 29-31")
cands = C.candidates(d)
check("ent-12 pairs the decision with Nina's constraint", len(cands) == 1 and cands[0]["kind"] == "decision_vs_constraint")
out = C.find_conflicts(d, Judge({"finance close": ("conflict", False)}))
check("ent-12 conflict returned with exact quotes", len(out) == 1 and out[0]["a"]["name"] == "Nina Albrecht" and "27th" in out[0]["a"]["text"] and "Denver" in out[0]["b"]["text"])
check("ent-12 shows the time", out and out[0]["a"]["when"] == "1:44" and out[0]["b"]["when"] == "4:22")
d["conflicts"] = out
check("status line counts it", "1 conflict to resolve" in S.status_line(d))
check("summary input carries the quotes", "CONFLICTS" in S.build_input(d) and "27th through November 3rd" in S.build_input(d))
check("count helper", C.count_conflicts(d) == 1)

# resolved later -> dropped; compatible -> dropped; tension kept but not counted
check("resolved_later conflict is dropped", C.find_conflicts(d, Judge({"finance close": ("conflict", True)})) == [])
check("compatible is dropped", C.find_conflicts(d, Judge({})) == [])
t = C.find_conflicts(d, Judge({"finance close": ("tension", False)}))
d2 = dict(d, conflicts=t)
check("tension is kept but not counted", len(t) == 1 and "conflict to resolve" not in S.status_line(d2))

# a bad model answer cannot invent a quote or crash
class Bad:
    def structured(self, **k): return dict(results=[dict(pair=99, verdict="conflict", why="x"), dict(pair="a"), "junk", dict(pair=0, verdict="nonsense")])
check("malformed judge output is ignored", C.find_conflicts(d, Bad()) == [])
class Boom:
    def structured(self, **k): raise RuntimeError("api down")
try:
    C.find_conflicts(d, Boom()); check("judge exception propagates to the caller (run_fast catches it)", False)
except RuntimeError:
    check("judge exception propagates to the caller (run_fast catches it)", True)

# ent-10: decision lists its own conditions; every pair the judge sees must be judged compatible, and none may come back
d10 = json.load(open(HERE.parent / "evals/outputs/ent-10-meeting-golive-decision.fast_result.json"))
j = Judge({})
check("ent-10 produces no conflict when the judge finds none", C.find_conflicts(d10, j) == [] and len(j.seen) >= 1)
check("ent-10 weak one-liners are not sent as constraints", not any(a.strip().endswith("?") or len(a.split()) < 6 for a in j.seen))

# ent-11: $60k -> $75k correction by the same person must never become a candidate
spans = teams((HERE / "sources/ent-11-meeting-vendor-pilot-checks.txt").read_text())
els = [dict(id="c1", kind="claim", title="pilot is $60k", status="active", created_span=next(s for s in spans if "call it $60k" in s["text"])["id"], spans=[], attrs={}),
       dict(id="c2", kind="claim", title="pilot is $75k", status="active", created_span=next(s for s in spans if "$75k, not $60k" in s["text"])["id"], spans=[], attrs={})]
d11 = dict(participants=[], span_msgs={}, elements=els, events=[], warnings=[],
           _enterprise=dict(source_type="transcript", speakers=sorted({s["author"] for s in spans}), msg_meta=[], spans=spans, classified=[]))
check("ent-11 acknowledged correction is not a candidate", C.candidates(d11) == [])

print("\n%d failed" % len(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
