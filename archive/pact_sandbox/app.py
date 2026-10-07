"""Run: streamlit run app.py"""
import json
import os

import streamlit as st

from core import LiveBackend, MockBackend, Simulation, Turn, load_scenario

st.set_page_config(page_title="Pact · Decision sandbox", page_icon="◉", layout="wide")
st.title("Pact · Decision sandbox")
st.caption("Six participants. One shared decision. Every state change has a source.")

with st.sidebar:
    mode = st.radio("Backend", ["Mock — no API calls", "Live — OpenAI"])
    model = st.text_input("Participant / engine model", os.getenv("PACT_MODEL", "gpt-4.1-mini"))
    research_model = st.text_input("Research model", os.getenv("PACT_RESEARCH_MODEL", "gpt-4.1-mini"))
    limit = st.slider("Turn limit", 6, 60, 24, 6)
    checks = st.slider("Maximum research requests", 0, 6, 3)
    if st.button("Start / reset run", type="primary"):
        try:
            backend = MockBackend() if mode.startswith("Mock") else LiveBackend(model, research_model, 3*limit+6)
            st.session_state.sim = Simulation(load_scenario(), backend, max_turns=limit, max_research=checks)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not start: {type(exc).__name__}. Check dependencies and OPENAI_API_KEY.")
    st.caption("Live mode sends public conversation to OpenAI. Each participant also sends its own private profile. Keys come only from the environment.")

if "sim" not in st.session_state:
    st.info("Choose Mock or Live and start a run. Mock is a scripted fixture, not an LLM experiment.")
    st.stop()

sim = st.session_state.sim
st.subheader(sim.scenario["question"])
if isinstance(sim.backend, MockBackend):
    st.warning("MOCK RUN: scripted participants and extraction; research is deliberately unverified.")
controls = st.columns([1, 1, 3])
steps = 0
if controls[0].button("Next message", disabled=sim.turns >= sim.max_turns):
    steps = 1
if controls[1].button("Next round (6)", disabled=sim.turns >= sim.max_turns):
    steps = min(6, sim.max_turns - sim.turns)
controls[2].caption(f"{sim.turns}/{sim.max_turns} turns · {sim.backend.calls} API calls · {sim.backend.tokens:,} tokens · {sim.research_count} checks")

left, right = st.columns([1.1, 1])
chat_slot = left.empty()
card_slot = right.empty()


def render():
    with chat_slot.container():
        st.subheader("Conversation")
        for msg in sim.messages[-40:]:
            with st.chat_message("assistant" if msg["actor"] == "Pact" else "user"):
                st.caption(f"{msg['actor']} · {msg['id']} · {msg['action']}")
                if msg["action"] == "interpretation":
                    st.json(json.loads(msg["text"]), expanded=True)
                else:
                    st.write(msg["text"])
        # Sources accompany Pact's research rather than hiding them in the state inspector.
        for ev in sim.evidence:
            if ev["sources"]:
                st.caption(f"Sources for {ev['id']} · {ev['retrieved_at']}")
                for source in ev["sources"]:
                    st.link_button(source["title"], source["url"])
    with card_slot.container():
        st.subheader("Where we are")
        st.caption("Recorded statements remain attributed; agreements require all six explicit confirmations.")
        labels = [("option", "Considering"), ("agreement", "Agreements / proposals"),
                  ("constraint", "Individual constraints"), ("preference", "Expressed preferences"),
                  ("claim", "Claims to check"), ("assumption", "Assumptions"), ("issue", "Open issues")]
        for kind, label in labels:
            items = [i for i in sim.items if i["kind"] == kind and i["active"]]
            if not items:
                continue
            st.markdown(f"**{label}**")
            for item in items:
                prefix = f"{item['person']}: " if item["person"] else ""
                st.write(f"{item['id']} · {prefix}{item['text']}")
                status = item["status"]
                if kind == "agreement":
                    status += f" · {len(item['confirmations'])}/6 confirmed"
                st.caption(f"{status} · source {item['source_message_id']}")
        if sim.evidence:
            st.markdown("**Research**")
            for ev in sim.evidence:
                with st.expander(f"{ev['id']} · {ev['status']}"):
                    st.write(ev["answer"])
                    st.caption(f"Retrieved {ev['retrieved_at']} · {ev['confidence']}")
                    for source in ev["sources"]:
                        st.link_button(source["title"], source["url"])
        if sim.interpretation:
            st.markdown("**Pact’s interpretation — awaiting human review**")
            st.json(sim.interpretation)
        st.markdown("**Facilitation consent**")
        st.write(" · ".join(f"{name}: {'opted in' if yes else 'not opted in'}" for name, yes in sim.consent.items()))
        with st.expander("Engine updates / guardrail decisions", expanded=True):
            for entry in sim.audit[-12:]:
                st.write(f"{entry['operation']} · {entry['status']}: {entry['detail']}")
        with st.expander("Inspect state and exact source quotes"):
            st.json(sim.public())


render()
for _ in range(steps):
    with st.spinner("Participant speaking; engine updating…"):
        try:
            sim.step()
        except Exception as exc:
            st.error(f"Step stopped: {type(exc).__name__}. Check model access, API key, or call budget. Export remains available.")
            break
    render()

with st.expander("Inject a message / test a correction"):
    with st.form("human_turn"):
        actor = st.selectbox("Speak as", sim.names)
        message = st.text_area("Message", max_chars=1600)
        action = st.selectbox("Explicit Pact request", ["say", "check", "resolve"])
        request = st.text_input("Claim / facilitation request", max_chars=500)
        consent = st.selectbox("Consent action", ["unchanged", "grant", "deny"])
        candidates = [i["id"] for i in sim.items if i["active"] and i["kind"] == "agreement"]
        confirms = st.multiselect("Explicitly confirm exact agreement wording", candidates)
        withdrawals = st.multiselect("Withdraw own item or own confirmation", [i["id"] for i in sim.items if i["active"]])
        if st.form_submit_button("Send", disabled=sim.turns >= sim.max_turns):
            try:
                sim.accept(actor, Turn(message=message, action=action, request=request,
                                      consent=consent, confirm_ids=confirms, withdraw_ids=withdrawals))
                st.rerun()
            except Exception as exc:
                st.error(f"Message rejected: {type(exc).__name__}.")

st.download_button("Export public run + engine audit (JSON)", json.dumps(sim.export(), indent=2),
                   file_name=f"pact-{sim.run_id}.json", mime="application/json")
with st.expander("Experiment controller: private profiles (not sent to Pact)"):
    st.json(sim.scenario["participants"])
st.caption("Learning means captured evaluation traces in this prototype. No automatic rule rewriting, training, or hidden influence scores.")
