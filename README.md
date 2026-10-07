# Pact MVP1 C2C

Pact is shared decision intelligence for groups. It maintains a source-linked decision card, checks a claim when asked, and helps move a discussion forward when invited.

This repository is currently focused on [`mvp1_c2c/`](mvp1_c2c/README.md), which holds two clearly separate things:

- **`mvp1_c2c/pact/`: the product.** Three functions: **observe** (fill the card from what people say), **check** (research claims on request), and **interpret** (explain where the decision stands). Python enforces provenance, permissions, and decision authority; models only propose.
- **`mvp1_c2c/simulation/`: the test harness.** One human administers and participates; configurable LLM participants join so Pact has a conversation to work on. It exists only to see whether Pact works.

## Start the experiment

```bash
pip install -r mvp1_c2c/requirements.txt
python -m uvicorn mvp1_c2c.server:app --port 8505
```

Set `ANTHROPIC_API_KEY` in the repository root `.env` or your shell before starting. Optional model settings are `ANTHROPIC_RESEARCH_MODEL` (Pact) and `ANTHROPIC_MODEL` (simulated participants). Session JSON files are written under `mvp1_c2c/runs/`.

Run the core checks from the repository root:

```bash
python3 -m pytest -q mvp1_c2c/tests
python3 -m compileall -q mvp1_c2c
```

Earlier work is preserved under `archive/`: the six-agent sandbox is in [`archive/pact_sandbox/`](archive/pact_sandbox/README.md), and the standalone observer plus its example cards are in [`archive/observer_prototype/`](archive/observer_prototype/). Neither is part of the active MVP.
