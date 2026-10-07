# Pact MVP1 C2C

MVP1 C2C is a local, controlled conversation experiment. One human is both the session admin and a participant. The admin configures one to five simulated participants, writes their roles and high-level guidance, chooses a bot-message limit, and can speak in the same conversation. Pact quietly observes messages, checks a selected claim only when asked, and offers facilitation only when invited.

This MVP is for exercising the product flow and card behavior. Simulated participants do not validate how real people behave or whether real groups trust Pact. Enterprise repositories, authentication, multi-user hosting, and external persistence are out of scope.

## Run it

```bash
cd mvp1_c2c
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

This MVP is LLM-driven: participant turns, card observation, claim checks, and Pact help all call Anthropic's Messages API. Put `ANTHROPIC_API_KEY` in the repository root `.env` (or export it in the shell) before starting. Optional model settings are `ANTHROPIC_MODEL` and `ANTHROPIC_RESEARCH_MODEL` (legacy `PACT_MODEL` and `PACT_RESEARCH_MODEL` are also read). API keys are sent only in the `x-api-key` header and are never saved in session files. Claim research uses Anthropic's hosted web search tool.

Sessions are saved as JSON under `runs/`. Each file contains the decision card and simulated participant profiles, but no provider credentials. Download a copy from the UI to share or archive a run.

## Experiment flow

1. Set a decision question, your participant name, simulated participant count, bot-message limit, and each bot's role, high-level guidance, and individual-mapping/sharing choices.
2. Send your own messages and advance one bot or a full round at a time.
3. Inspect the source-linked decision card and audit trail as Pact observes.
4. Select a recorded claim and explicitly request a check; ask Pact for help when you want an interpretation.
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

Tests inject fixed LLM-shaped responses and do not contact Anthropic. The fake exists only in the test suite; the app has no regex or scripted conversation backend.
