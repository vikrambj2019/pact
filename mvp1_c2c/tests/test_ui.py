"""Execute the browser renderer with current card data in a small DOM stub."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fakes import FakePactLLM, act, claim
from mvp1_c2c.pact import PactSession

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="Node.js is needed for browser renderer tests")
PAGE = Path(__file__).resolve().parents[1] / "static" / "index.html"


def render(card):
    script = r'''
const fs = require("fs"), vm = require("vm");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const code = fs.readFileSync(input.page, "utf8").match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace(/\(async function init\(\) \{[\s\S]*?\n\}\)\(\);/, ""); // render-only test, no setup fetch
const elements = {};
const context = vm.createContext({window: {}, document: {getElementById(id) {
  return elements[id] ||= {innerHTML: "", textContent: "", disabled: false};
}}});
vm.runInContext(code, context);
context.input = input;
vm.runInContext("renderAll({card:input.card, simulation:{bot_turns:0,max_bot_turns:5}, calls:0})", context);
process.stdout.write(elements["card-body"].innerHTML);
'''
    result = subprocess.run([NODE, "-e", script], input=json.dumps({"page": str(PAGE), "card": card}),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_browser_shows_current_options_verdicts_historical_reports_and_decision_control():
    llm = FakePactLLM({"Banff": [act("add_option", "Banff", text="Banff")],
                       "Fee is $20": [claim("Fee is $20")],
                       "Fee is $200": [act("correct_claim", "Fee is $200", text="Fee is $200", target_id="claim_1")]})
    session = PactSession.create("Where?", "Sam", [], llm)
    session.add_admin_message("Banff")
    session.add_admin_message("Fee is $20")
    session.check_claim("claim_1")
    session.add_admin_message("Fee is $200")
    html = render(session.card)
    assert "Options" in html and "Select option" in html and "Banff" in html
    assert "Fee is $200" in html and "not checked" in html
    assert "Historical" in html and "Fixture: no external research." in html
    assert "https://example.gov/a" in html
    assert "Required rationale" in html
    assert "undefined" not in html


def test_browser_hides_unshared_preferences_and_marks_old_summary():
    members = [{"id": "alex", "name": "Alex", "mapping_allowed": True, "share_allowed": False}]
    llm = FakePactLLM({"private preference": [act("set_preference", "private preference", stance="private preference")]})
    session = PactSession.create("Where?", "Sam", members, llm)
    session.add_message("alex", "private preference", "human")
    session.interpret()
    session.add_admin_message("Later message")
    html = render(session.card)
    assert "private preference" not in html
    assert "ask Pact for an update" in html
