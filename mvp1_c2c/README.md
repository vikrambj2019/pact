# Pact MVP1 C2C

MVP1 C2C is a local, controlled conversation experiment. One human is both the session admin and a participant. The admin configures one to five simulated participants, writes their roles and high-level guidance, chooses a bot-message limit, and can speak in the same conversation. Pact quietly observes messages, checks a selected claim only when asked, and offers facilitation only when invited.

This MVP is for exercising the product flow and card behavior. Simulated participants do not validate how real people behave or whether real groups trust Pact. Enterprise repositories, authentication, multi-user hosting, and external persistence are out of scope.

## Pact vs. the simulation

The code is split so it is always clear what is the product and what is only test scaffolding:

```
mvp1_c2c/
  pact/            THE PRODUCT. Knows nothing about simulated participants.
    observe.py       1. Observe   — every message → source-quoted card changes
    check.py         2. Check     — on request: one claim, one person's claims, or all claims
    interpret.py     3. Interpret — on request: where the decision stands and the next step
    card.py          the decision card and the rules every change must pass
    session.py       PactSession: the three functions + admin authority (confirm, decide)
    commands.py      routes "pact, ..." messages to check or interpret
  simulation/      TEST HARNESS. LLM-played participants that give Pact a conversation to observe.
  server.py        FastAPI: Pact routes and Simulation routes, kept in separate sections
  static/          browser UI
```

`simulation` imports `pact`; `pact` never imports `simulation`. Simulation state (profiles, private guidance, round-robin order, bot-message limit) is saved beside the card under `"simulation"`, never inside it.

### What Pact checks

A claim is checkable only when it is a factual statement about the outside world that public web sources can confirm or contradict (prices, distances, dates, weather, closures, permits, rules, availability). Opinions, predictions about the group, and facts about a participant themselves are recorded but never researched. The observer marks each claim `checkable`; the definition lives in `pact/check.py`. To restrict which websites a check may search, set `PACT_CHECK_DOMAINS` (comma-separated, e.g. `nps.gov,weather.gov`); unset means any site.

## Run it

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r mvp1_c2c/requirements.txt
python -m uvicorn mvp1_c2c.server:app --port 8505
```

Open http://localhost:8505. Put `ANTHROPIC_API_KEY` in the repository root `.env` (or export it) before starting. `ANTHROPIC_RESEARCH_MODEL` sets Pact's model (observe, check, interpret); `ANTHROPIC_MODEL` sets the simulated participants' model. Legacy `PACT_RESEARCH_MODEL` and `PACT_MODEL` are also read. API keys are sent only in the `x-api-key` header and are never saved in session files. Claim checks use Anthropic's hosted web search tool.

Sessions are saved as JSON under `runs/`, with no provider credentials. Sessions saved before the Pact/simulation split load and are migrated in memory.

## Experiment flow

1. Set a decision question, your participant name, simulated participant count, bot-message limit, and each bot's role, high-level guidance, and individual-mapping/sharing choices.
2. Send your own messages and advance one bot or a full round at a time.
3. Inspect the source-linked decision card and audit trail as Pact observes.
4. Check a claim, one person's claims, or all claims (button or "pact, check Alex's claims"); ask Pact to interpret when you want to know where things stand.
5. If a proposal is recorded, explicitly record a decision and rationale. The app keeps a snapshot with the context available at that time.

Bot messages are round-robin. The bot-message limit excludes your messages. High-level guidance is sent privately to its assigned simulated participant. Permission and decision controls are handled by Python and the UI, not by model output.

## Data and guardrails

- Card changes must cite exact text from a recorded source message.
- Individual positions must be attributed to their speaker and are only recorded with mapping permission. Shared display also checks the participant's sharing permission.
- Ambiguous human interpretations stay outside confirmed card state until explicitly confirmed. Ambiguous simulated-participant interpretations are skipped and recorded in the audit because simulated participants cannot confirm items in this MVP.
- Models cannot write decision, permission, participant, or history fields. Only the admin UI can record the session decision.
- Claim checks record reports and citations as evidence; they do not set agreement or establish truth.
- Decision snapshots preserve the selected proposal, rationale, current evidence/objections, and source message IDs.

## Verify

```bash
python -m pytest -q mvp1_c2c/tests
python -m compileall -q mvp1_c2c
```

`tests/test_pact.py` exercises Pact alone, organized by observe / check / interpret; `tests/test_simulation.py` covers the harness. Both use fixed LLM-shaped responses from `tests/fakes.py` and never contact Anthropic.
