"""Reviewer boundary regressions: coverage, failure, and publication ordering."""
import asyncio
import json
import unittest
from pathlib import Path
from load_pipeline import reviewer, extractor


class PassingLLM:
    def __init__(self):
        self.requests = []

    async def extract(self, system, user, tool, **kwargs):
        payload = json.loads(user)
        self.requests.append(payload)
        return {"verdicts": [{"key": e["key"], "verdict": "pass", "reason": "Supported by cited source"}
                             for e in payload["elements"]], "consistency": "pass", "issues": []}


class ReviewerTests(unittest.TestCase):
    def test_all_sections_and_every_element_are_sent_with_sources(self):
        draft = {s: [{"text": "one"}, {"text": "two"}] for s in reviewer.SECTIONS}
        llm = PassingLLM()
        report = asyncio.run(reviewer.review_extraction(draft, [{"id": "source"}], llm))
        keys = [e["key"] for p in llm.requests for e in p["elements"]]
        self.assertEqual(set(keys), {f"{s}:{i}" for s in reviewer.SECTIONS for i in range(2)})
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(report["coverage_complete"])
        self.assertTrue(all(p["source_units"] == [{"id": "source"}] for p in llm.requests))

    def test_missing_duplicate_unknown_and_malformed_verdicts_fail_closed(self):
        row = {"key": "actions:0", "verdict": "pass", "reason": "supported"}
        for rows in ([], [row, row], [{**row, "key": "other"}], [{**row, "reason": ""}], [{**row, "verdict": "maybe"}]):
            with self.subTest(rows=rows), self.assertRaises(reviewer.ReviewError):
                reviewer.validate_verdicts({"verdicts": rows, "consistency": "pass", "issues": []}, ["actions:0"])

    def test_unsupported_deadline_or_global_contradiction_blocks(self):
        for response in (
            {"verdicts": [{"key": "actions:0", "verdict": "reject", "reason": "Dev never accepted Nov 6 as completion date"}], "consistency": "pass", "issues": []},
            {"verdicts": [{"key": "actions:0", "verdict": "pass", "reason": "supported"}], "consistency": "reject", "issues": ["Funding escalation misrepresented as approval"]},
        ):
            with self.subTest(response=response), self.assertRaises(reviewer.ReviewError):
                reviewer.validate_verdicts(response, ["actions:0"])

    def test_transport_failure_and_empty_draft_still_require_review(self):
        class Offline:
            async def extract(self, *a, **k):
                raise RuntimeError("offline")
        with self.assertRaisesRegex(reviewer.ReviewError, "publication blocked"):
            asyncio.run(reviewer.review_extraction({}, [], Offline()))
        llm = PassingLLM()
        asyncio.run(reviewer.review_extraction({}, [], llm))
        self.assertEqual(len(llm.requests), 1)

    def test_summary_reviews_each_nonempty_line_and_complete_json(self):
        llm = PassingLLM()
        draft = {"actions": [{"text": "Fix #3", "due": None}]}
        summary = "Portal launch\n\nNEEDS ATTENTION\nFinding #3 remains open"
        report = asyncio.run(reviewer.review_summary(summary, draft, [], llm))
        self.assertEqual(report["reviewed_element_count"], 3)
        self.assertEqual(llm.requests[0]["complete_draft"]["structured"], draft)
        self.assertEqual(llm.requests[0]["complete_draft"]["executive_summary"], summary)

    def test_failed_review_feedback_is_used_then_retry_is_reviewed(self):
        class RetryLLM:
            def __init__(self):
                self.reviews = 0
                self.prompts = []
            async def extract(self, system, user, tool, **kw):
                if tool["name"] != "review_elements":
                    self.prompts.append(user)
                    return {}
                self.reviews += 1
                return {"verdicts": [], "consistency": "reject" if self.reviews == 1 else "pass",
                        "issues": ["Missing unresolved security gate"] if self.reviews == 1 else []}
        llm = RetryLLM()
        result = asyncio.run(extractor.extract_chunk([], llm))
        self.assertEqual(llm.reviews, 2)
        self.assertIn("Missing unresolved security gate", llm.prompts[2])
        self.assertEqual(result["_review"]["status"], "passed")

    def test_final_summary_review_precedes_revision_persistence(self):
        import ast
        tree = ast.parse((Path(__file__).parents[1] / "fixed/synthesizer.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_synthesize_brief")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        review = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "review_summary")
        revision = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "BriefRevision")
        self.assertLess(review.lineno, revision.lineno)


if __name__ == "__main__":
    unittest.main()
