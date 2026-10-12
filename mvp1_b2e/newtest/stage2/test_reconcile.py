"""Offline test for reconcile.py + score_reconcile.py with a scripted model.  Run: python test_reconcile.py"""
import json
import re
import tempfile
from pathlib import Path

from reconcile import changes, dump, matrix, reconcile
from score_reconcile import report, score

def S(i, author, msg, text, *labels):
    return {"span_id": f"s{i}", "author": author, "msg": msg, "ts": f"2026-10-26T09:{i:02d}:00",
            "text": text, "status": "active", "labels": [{"type": l, "gist": ""} for l in labels]}

ROWS = [
    S(1, "Sofia", "m1", "land the end-state data platform", "candidate_question"),
    S(2, "Colleen", "m2", "Databricks, obviously.", "option", "position"),
    S(3, "Raj", "m3", "Snowflake.", "option", "position"),
    S(4, "Dmitri", "m4", "CFO wants a 15% run-cost reduction", "constraint"),
    S(5, "Raj", "m5", "I could live with Databricks IF the tooling covers our 40 procs", "position"),
    S(6, "Colleen", "m6", "Great, so Raj is fine with Databricks.", "claim"),
    S(7, "Sofia", "m7", "A is Databricks, B is Snowflake.", "process"),
    S(8, "Raj", "m8", "B for now.", "position"),
    S(9, "Dmitri", "m9", "Correction: the CFO target is 10%, not 15%.", "constraint"),
    S(10, "Colleen", "m10", "Databricks is the obvious end state", "position"),
    S(11, "Raj", "m11", "If you lend two engineers, A works for me and I withdraw B.", "position"),
    S(12, "Sofia", "m12", "Thanks", "irrelevant"),
    S(13, "Sofia", "m13", "Direction: Databricks as the end state.", "decision"),
    # inferred-stance traps
    S(14, "Dmitri", "m14", "pulled the Snowflake contract: renews Feb 28", "claim"),           # Nora pattern
    S(15, "Sofia", "m15", "the Q2 cutover works for me", "position"),                         # Priya pattern
    S(16, "Dmitri", "m16", "+1", "position"),                                                 # short reply: counts
    S(17, "Sofia", "m17", "Databricks with Halden engineers could work too", "option"),       # variant -> alias
    S(18, "Dmitri", "m18", "Snowflake is off the table for me", "position"),                 # model forgets it
    S(19, "Imani", "m19", "or we just keep querying both Databricks and Snowflake", "option"), # NOT a variant
]

SCRIPT = {
    "s1": [{"op": "create", "ref": "q", "kind": "question", "title": "End-state data platform", "span": "s1"}],
    "s2": [{"op": "create", "ref": "db", "kind": "option", "title": "Databricks", "question": "q1", "span": "s2"},
           {"op": "stance", "option": "db", "stance": "proposed", "span": "s2"}],
    "s3": [{"op": "create", "ref": "sf", "kind": "option", "title": "Snowflake", "question": "q1", "span": "s3"},
           {"op": "stance", "option": "sf", "stance": "proposed", "span": "s3"}],
    "s4": [{"op": "create", "kind": "constraint", "title": "CFO 15% run-cost cut", "span": "s4"}],
    "s5": [{"op": "stance", "option": "opt1", "stance": "conditional", "conditions": "tooling covers 40 procs", "span": "s5"}],
    # the model misbehaves: tries to give Raj a stance from Colleen's span -> participant is forced to Colleen
    "s6": [{"op": "create", "kind": "claim", "title": "Raj is fine with Databricks", "claim_kind": "consensus",
            "asserts_stances": [{"participant": "Raj", "option": "opt1", "stance": "support"}], "span": "s6"},
           {"op": "stance", "option": "opt1", "stance": "support", "span": "s6"}],
    "s7": [{"op": "alias", "element": "opt1", "alias": "A", "span": "s7"},
           {"op": "alias", "element": "opt2", "alias": "B", "span": "s7"}],
    "s8": [{"op": "stance", "option": "opt2", "stance": "support", "span": "s8"}],
    "s9": [{"op": "create", "ref": "c10", "kind": "constraint", "title": "CFO 10% run-cost cut", "span": "s9"},
           {"op": "supersede", "old": "con1", "new": "c10", "span": "s9"}],
    # duplicate create must be redirected to opt1
    "s10": [{"op": "create", "kind": "option", "title": "databricks", "span": "s10"},
            {"op": "stance", "option": "opt1", "stance": "support", "span": "s10"}],
    "s11": [{"op": "stance", "option": "opt1", "stance": "conditional", "conditions": "two engineers lent", "span": "s11"},
            {"op": "stance", "option": "opt2", "stance": "withdrawn", "span": "s11"},
            {"op": "status", "element": "opt2", "status": "dropped", "span": "s11", "because": ["opt1"]},
            {"op": "status", "element": "opt1", "status": "chosen", "span": "s11"}],   # illegal: must use decide
    "s13": [{"op": "decide", "question": "q1", "chosen": ["opt1"], "ruled_out": ["opt2"], "span": "s13"}],
    "s14": [{"op": "stance", "option": "opt2", "stance": "support", "span": "s14"}],
    "s15": [{"op": "stance", "option": "opt2", "stance": "oppose", "span": "s15"}],
    "s16": [{"op": "stance", "option": "opt1", "stance": "support", "span": "s16"}],
    "s17": [{"op": "create", "kind": "option", "title": "Databricks with Halden engineers", "question": "q1", "span": "s17"}],
    "s18": [],                                                    # first pass: missed entirely
    "s19": [{"op": "create", "kind": "option", "title": "Keep both Databricks and Snowflake", "question": "q1", "span": "s19"}],
}
FOLLOWUP = {"s18": [{"op": "stance", "option": "opt2", "stance": "oppose", "span": "s18"}]}


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.seen = []
        self.followups = []

    def structured(self, *, system, user, tool_name, schema, max_tokens=2048):
        self.calls += 1
        new = user.split("NEW SPANS:")[1]
        ids = re.findall(r"^\[(s\d+)\]", new, re.M)
        if "MISSED STANCES" in user:
            self.followups += ids
            return {"ops": [op for sid in ids for op in FOLLOWUP.get(sid, [])]}
        self.seen += ids
        return {"ops": [op for sid in ids for op in SCRIPT.get(sid, [])]}


GOLD = {
    "participants": ["Sofia", "Colleen", "Raj", "Dmitri"],
    "questions": [{"key": "platform", "match": ["platform"], "status": ["decided"]}],
    "options": [
        {"key": "databricks", "match": ["databricks", "^a$"], "exclude": ["both"], "status": ["chosen"],
         "stances": {"Colleen": ["support"], "Raj": ["conditional"], "Dmitri": ["support"]}},
        {"key": "snowflake", "match": ["snowflake", "^b$"], "exclude": ["both"], "status": ["dropped"],
         "stances": {"Raj": ["withdrawn"], "Dmitri": ["oppose"]}},
    ],
    "elements": [{"key": "cfo15", "msg": "m4", "kinds": ["constraint"], "status": ["superseded"]}],
    "histories": [{"option": "snowflake", "participant": "Raj", "sequence": [["support"], ["withdrawn"]]},
                  {"option": "databricks", "participant": "Raj", "sequence": [["conditional"]]}],
    "flags": [{"type": "consensus_conflict", "msg": "m6"}],
}


def test():
    llm = FakeLLM()
    people = ["Sofia", "Colleen", "Raj", "Dmitri"]
    reg = reconcile(ROWS, people, llm, batch=4, log=None)

    assert "s12" not in llm.seen, "irrelevant-only span should not be sent"
    assert llm.calls == 6                                         # 18 content spans / 4 + 1 follow-up
    assert llm.followups == ["s18"], llm.followups                # only the stance-less position span
    assert reg.stance_now("opt1", "Raj") == "conditional", "Colleen's claim must not move Raj"
    assert reg.stance_now("opt1", "Colleen") == "support"         # from her own s10, not the s6 claim
    inf = {(e.span_id, e.participant) for e in reg.events if e.kind == "stance" and e.payload.get("inferred")}
    assert ("s6", "Colleen") in inf, "stance from a claim span must be inferred"
    assert ("s14", "Dmitri") in inf, "stance from an action span must be inferred (Nora pattern)"
    assert ("s15", "Sofia") in inf, "stance on an option the span never names must be inferred (Priya pattern)"
    assert ("s16", "Dmitri") not in inf, "short reply must count"
    assert reg.stance_now("opt2", "Sofia") is None                 # her s15 stance was inferred
    assert reg.stance_now("opt1", "Dmitri") == "support"
    opts = {o.title for o in reg.live("option")}
    assert opts == {"Databricks", "Snowflake", "Keep both Databricks and Snowflake"}, opts
    assert reg.get("opt1").aliases == ["A", "Databricks with Halden engineers"], reg.get("opt1").aliases
    assert reg.stance_now("opt2", "Dmitri") == "oppose", "follow-up pass must recover the missed stance"
    assert reg.get("con1").status == "superseded" and reg.get("con1").superseded_by == "con2"
    assert reg.get("opt1").status == "chosen" and reg.get("opt2").status == "dropped"
    assert reg.get("q1").status == "decided"
    assert any("use decide" in w for w in reg.warnings)
    assert any("duplicate option" in w for w in reg.warnings)
    flags = [e for e in reg.events if e.kind == "flag"]
    assert len(flags) == 1 and flags[0].payload["mismatches"][0]["actual"] == "conditional"

    print(matrix(reg, people))
    print("\nWHAT SHIFTED")
    print("\n".join(changes(reg)))

    with tempfile.TemporaryDirectory() as tmp:
        out, gold = Path(tmp, "x.reconcile.json"), Path(tmp, "x.state.json")
        out.write_text(json.dumps(dump(reg, people, "test", "fake")))
        gold.write_text(json.dumps(GOLD))
        R, errs, dup, d = score(str(out), str(gold))
        print()
        report(R, errs, dup, d, True)
        assert all(ok == n for ok, n in R.values()), errs
        assert not dup, dup
    print("\nall checks passed")


if __name__ == "__main__":
    test()
