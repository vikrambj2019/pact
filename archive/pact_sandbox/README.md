# Pact: six-agent decision sandbox

A small Python experiment: six prompted participants talk, Pact keeps a source-backed decision card, and participants can explicitly ask Pact to check a factual claim or interpret what is blocking their decision. Streamlit displays conversation on the left and state / engine updates on the right.

## Run

Requires Python 3.10+.

```bash
cd archive/pact_sandbox
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Start with **Mock**, then press **Next round (6)**. Mock uses deterministic scripted fixtures to exercise the UI; it is not an experiment with actual LLM participants and performs no research.

For six real model-driven participants, set a key before launching:

```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

Select **Live — OpenAI** and start a new run. Model IDs can be changed in the sidebar or with `PACT_MODEL` and `PACT_RESEARCH_MODEL`. Defaults are `gpt-4.1-mini`; your account must have access to the configured models, Structured Outputs, and web search. OpenAI API billing is separate from a ChatGPT subscription. Never commit keys or put them in scenario files.

Headless runs:

```bash
python core.py --mode mock --turns 18
python core.py --mode live --turns 24 --model gpt-4.1-mini
```

Run exports go to `runs/` beside the code, or the directory specified by `--output`. In the UI, download the JSON export before resetting or closing the session; UI runs are held in session memory.

## Small architecture

- `scenario.json`: six private profiles, goals, preferences and constraints. Edit these and the question to create another experiment. Exactly six unique names are required.
- `core.py`: Pydantic contracts, prompts, SDK backend, scripted backend, deterministic state reducer, consent and audit log.
- `app.py`: Streamlit conversation, live card, consent view, state inspector, message injection and export.
- `tests/test_core.py`: state, consent, provenance, SDK contract and UI tests.

One simulation step:

1. Pick the next participant from a seeded shuffled round. Each gets its own private profile and the public state / recent conversation.
2. Generate a structured turn containing its natural-language message and explicit actions.
3. Append the message. Process explicit consent, confirmations and withdrawals in Python.
4. Invoke a separate extraction prompt to propose additions. Validate quotations, attribution and relationships before applying them.
5. Only if explicitly requested, run a factual check or consent-gated interpretation. A resolve request never silently triggers browsing.
6. Render the new card and append auditable events.

Pact never receives the other participants' private profiles. The experiment controller can inspect them in a separate expander; they are not included in public run exports. Separate API requests use `store=False`, without shared threads. Participants are sequential, not concurrent, so messages and state have a reproducible order. Reusing a seed reproduces speaker order, not live model outputs.

## Guardrails enforced by code

- Only the selected actor can submit its turn; models cannot impersonate another participant through structured actor fields.
- Self-stated preferences and constraints are attributed to their source speaker. Exact nonempty source quotes are required for all proposed additions. References must point to active items.
- Repeated identical items are ignored; repeating a claim never raises confidence.
- An agreement starts as **Proposed**. It becomes **Agreed** only after six distinct explicit `confirm_ids` actions. Text-only approval, silence and conditional support do not count.
- A person can withdraw its own earlier item or confirmation; it cannot delete another person's statement. Withdrawing a confirmation reopens an agreement.
- All participants initially have consent **off**. Facilitation requires six explicit opt-ins. Revocation blocks future facilitation and clears the current interpretation. Historical messages remain in the audit; revocation is not a deletion mechanism.
- Research only runs for a nonempty explicit `check` action, with a per-run research cap. Models otherwise have no browsing, shell, filesystem or arbitrary tool access.
- Research provenance comes from completed SDK web-search output and its URL annotations, never generated URL strings. Retrieved timestamps and citation spans are retained. Sourced research is not labeled verified. Mock research is always unverified and labeled simulated.
- Pact interpretations require existing message references and are labeled interpretations, never agreements. Any new participant turn invalidates the last displayed interpretation.
- Typed outputs reject unexpected keys. Failed extraction keeps the message and prior state; failed research never fabricates evidence. Turn / API-request / research limits bound runs. SDK calls have timeouts and one retry.

## What "learning" means here

The engine updates decision state during the run and captures proposed additions, rejected changes, confirmations, withdrawals, consent events, research and interventions. Exported traces support evaluation and subsequent prompt / rule improvements. This version does **not** train models or automatically rewrite governing rules. No hidden credibility or influence scores exist.

Useful next experiments are matched scenarios with Pact, without Pact, and with a summary baseline. Evaluate fidelity, missed constraints, unsupported agreements, preserved objections and feasible outcomes separately from satisfaction. Those comparison arms and a trained intervention policy are not implemented in this small first version.

## Limits worth understanding

Exact quotation and attribution checks establish provenance, not semantic correctness. A model may still paraphrase incorrectly, misclassify a statement, emit an inappropriate confirmation or overinterpret a cited message. Structured confirmations are explicit simulated agent actions; they are not verified human consent. The prompts constrain decision scope and neutral language, but Python does not fully verify relevance, neutrality or factual entailment. Web citations establish source provenance, not truth. Do not interpret simulation success as evidence of human adoption or neutrality.

Corrections use explicit withdrawal plus a new statement; the engine does not automatically supersede conflicting statements, reopen agreements because of new evidence, or prove conditional feasibility. Dependencies are recorded as links, without a general constraint solver. Private preference collection, multi-decision boundaries, late joiners, persistent shared hosting and Telegram integration are deliberately absent.

Public context uses the last 40 messages plus current state and evidence; extraction always sees the new message. Longer histories can lose conversational nuances. Keys are not exported, but anything a participant says publicly may contain information it chose to reveal. Local exports include all public messages and the audit. This is a single-user local sandbox; the controller can inject a message as any participant for testing and is not an authentication system.

## Verification

```bash
pip install pytest
python -m pytest -q
```

Tests exercise consent gating / revocation, six-person confirmation / withdrawal, unsupported attribution and quotes, conditional language, research limits and provenance, context separation, failure behavior, an 18-turn mock run, the real SDK serialization contract through a mocked HTTP transport, and Streamlit's UI testing harness. Live API calls require your own key; no live research or participant behavior has been validated in this delivery.

Implementation references:

- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/tools-web-search
