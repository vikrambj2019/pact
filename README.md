# Pact MVP1 C2C

Pact is shared decision intelligence for groups. It maintains a source-linked decision card, checks a claim when asked, and helps move a discussion forward when invited.

This repository is currently focused on the local simulated-group experiment in [`mvp1_c2c/`](mvp1_c2c/README.md). One human participates and administers the session; configurable LLM participants join the conversation. Participant turns, card observation, claim checks, and Pact help all use the LLM. Python enforces provenance, permissions, and decision authority.

## Start the experiment

```bash
cd mvp1_c2c
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` in the repository root `.env` or your shell before starting. Optional model settings are `ANTHROPIC_MODEL` and `ANTHROPIC_RESEARCH_MODEL`. Session JSON files are written under `mvp1_c2c/runs/`.

Run the core checks from the repository root:

```bash
python3 -m pytest -q mvp1_c2c/tests
python3 -m compileall -q mvp1_c2c
```

Earlier work is preserved under `archive/`: the six-agent sandbox is in [`archive/pact_sandbox/`](archive/pact_sandbox/README.md), and the standalone observer plus its example cards are in [`archive/observer_prototype/`](archive/observer_prototype/). Neither is part of the active MVP.
