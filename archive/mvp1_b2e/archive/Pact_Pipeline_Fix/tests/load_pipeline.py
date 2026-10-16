"""Load pure pipeline functions without the missing surrounding application.

The real application should import its modules normally. This loader is only
for standalone regression/live extraction tests of the eight supplied files.
"""
import ast
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "fixed"

def load(name, extra=None):
    tree = ast.parse((ROOT / (name + ".py")).read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("sqlalchemy")))]
    module = types.ModuleType(name)
    module.__dict__.update({"AsyncSession": object, **(extra or {})})
    exec(compile(tree, str(ROOT / (name + ".py")), "exec"), module.__dict__)
    return module

reviewer = load("reviewer")
extractor = load("extractor", {"review_extraction": reviewer.review_extraction})
merger = load("merger")
chunker = load("chunker")
dedup = load("dedup")
parser = load("parser")
synthesizer = load("synthesizer", {
    "review_extraction": reviewer.review_extraction, "review_summary": reviewer.review_summary,
    "quote_in_text": extractor.quote_in_text, "_norm_text": merger._norm_text,
    "_same_option": merger._same_option, "_similar": merger._similar,
})
