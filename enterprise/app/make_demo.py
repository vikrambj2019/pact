"""Build static/demo/data.json: recorded Pact runs for the replay (static) site.

    python3 enterprise/app/make_demo.py

Only synthetic sources go in (ent-*). Never add real-* or long-* files: this data is published as a public static site.
A sample is included only if both its source and a run_fast output exist. Re-run after producing new outputs
(e.g. ent-11, ent-12) with run_fast.py.
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
EV = H.parent / "evals"
SAMPLES = [
    ("ent-12-meeting-offsite-ignored-point", "Offsite: one objection nobody answers", "transcript", "Five people pick a city and dates. One point is raised twice and never answered."),
    ("ent-11-meeting-vendor-pilot-checks", "Vendor pilot: conditions and a split decision", "transcript", "Two questions at once, conditional agreement and a corrected figure."),
    ("ent-10-meeting-golive-decision", "Go-live date: a decision with conditions", "transcript", "A date is chosen subject to a bug fix, an access list and coverage."),
    ("ent-09-email-offsite-reversal-forward", "Email: an announced reversal that reached only some people", "email", "A reversal is announced to part of the thread."),
    ("ent-08-email-vendor-private-note", "Email: a private note in a reply-all chain", "email", "One person's private concern is quoted to everyone."),
]
KEEP = ("participants", "summary", "elements", "events", "warnings", "conflicts", "source", "model")

out = []
for sid, title, kind, blurb in SAMPLES:
    src, res = EV / "sources" / f"{sid}.txt", EV / "outputs" / f"{sid}.fast_result.json"
    if not (src.exists() and res.exists()):
        print("skip (no source or output):", sid)
        continue
    r = json.loads(res.read_text())
    out.append(dict(id=sid, title=title, blurb=blurb, type=kind, text=src.read_text(),
                    duration=r.get("duration_seconds") or 0, result={k: r[k] for k in KEEP if k in r}))
    print("added:", sid, "(has logic check)" if "conflicts" in r else "(no logic check in this output)")
dest = H / "static" / "demo"
dest.mkdir(exist_ok=True)
(dest / "data.json").write_text(json.dumps({"samples": out}, ensure_ascii=False))
print(f"wrote {dest / 'data.json'}  {(dest / 'data.json').stat().st_size // 1024} KB, {len(out)} samples")
