"""Offline wiring test: scripted fake LLM, no API calls.  Run: python test_pipeline.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

from classify import classify
from models import ElementType, RawMessage, SpanStatus, Transcript
from normalize import coalesce, normalize

T0 = datetime(2026, 10, 11, 9, 0)
def m(i, author, sec, text):
    return RawMessage(id=f"m{i}", author=author, ts=T0 + timedelta(seconds=sec), text=text)

TRANSCRIPT = Transcript(messages=[
    m(1, "Rosa",   0,   "where should we do the hike this month?"),
    m(2, "Vikram", 30,  "yeah I'll do 24th"),
    m(3, "Vikram", 45,  "oh sorry, 24th doesn't work, I have a thing"),
    m(4, "Vikram", 50,  "I can do the 5th"),
    m(5, "Priya",  200, "lol nice"),
    m(6, "Priya",  230, "I can't do the 24th but I can do the 5th. Diablo parking is a nightmare on weekends though"),
])


class FakeLLM:
    """Returns canned outputs keyed on the tool name + a substring of the prompt."""
    def structured(self, *, system, user, tool_name, schema, max_tokens=2048):
        if tool_name == "segment":
            burst = user.split("BURST TO SEGMENT")[1]
            if "Rosa" in burst.splitlines()[0]:
                return {"spans": [{"message_index": 0, "text": "where should we do the hike this month?", "status": "active"}]}
            if "Vikram" in burst.splitlines()[0]:
                return {"spans": [
                    {"message_index": 0, "text": "I'll do 24th", "status": "active"},
                    {"message_index": 1, "text": "24th doesn't work, I have a thing", "status": "active", "corrects_index": 0},
                    {"message_index": 2, "text": "I can do the 5th", "status": "active", "context_note": "5th = proposed hike date"},
                ]}
            return {"spans": [
                {"message_index": 0, "text": "lol nice", "status": "active"},
                {"message_index": 1, "text": "I can't do the 24th", "status": "active"},
                {"message_index": 1, "text": "I can do the 5th", "status": "active"},
                {"message_index": 1, "text": "Diablo parking is a nightmare on weekends", "status": "active"},
            ]}
        target = user.split(">>> SPAN TO LABEL:\n")[1].split("\n")[0]
        table = {
            "hike this month": [("candidate_question", .95)],
            "I'll do 24th": [("position", .6)],
            "doesn't work": [("position", .9), ("constraint", .7)],
            "can't do the 24th": [("position", .9), ("constraint", .7)],
            "can do the 5th": [("position", .9), ("option", .8)],
            "parking": [("objection", .85), ("claim", .7), ("irrelevant", .2)],
            "lol": [("irrelevant", .95)],
        }
        for k, labs in table.items():
            if k in target:
                return {"labels": [{"type": t, "confidence": c, "gist": k} for t, c in labs],
                        "rationale": "fake"}
        return {"labels": [{"type": "irrelevant", "confidence": .5, "gist": ""}], "rationale": "fake"}


def test():
    bursts = coalesce(TRANSCRIPT.messages)
    assert [b.author for b in bursts] == ["Rosa", "Vikram", "Priya"], bursts
    assert bursts[1].message_ids == ["m2", "m3", "m4"]

    spans = normalize(TRANSCRIPT, FakeLLM())
    assert len(spans) == 8
    assert all(s.verified for s in spans)
    v1 = spans[1]
    assert v1.status == SpanStatus.self_corrected and v1.superseded_by == spans[2].id
    assert spans[2].ts == T0 + timedelta(seconds=45)          # ts of its own message
    assert spans[6].text == "I can do the 5th" and spans[6].author == "Priya"

    out = classify(spans, FakeLLM())
    assert len(out) == len(spans)
    assert out[4].is_irrelevant                               # "lol nice"
    park = out[7]
    assert {l.type for l in park.labels} == {ElementType.objection, ElementType.claim}

    for cs in out:
        flag = " (self-corrected)" if cs.span.status == SpanStatus.self_corrected else ""
        tags = ", ".join(f"{l.type.value}:{l.confidence:.2f}" for l in cs.labels)
        print(f"{cs.span.id:7} {cs.span.author:7} {cs.span.text!r:46} -> {tags}{flag}")
    print("\nall checks passed")


if __name__ == "__main__":
    test()
