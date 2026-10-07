"""Run with: streamlit run mvp1_c2c/app.py"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

try:
    from mvp1_c2c.core import AnthropicBackend, Simulation, public_card
except ImportError:
    from core import AnthropicBackend, Simulation, public_card

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)

st.set_page_config(page_title="Pact · MVP1 C2C", page_icon="◉", layout="wide")
st.title("Pact · conversation experiment")
st.caption("A local experiment with simulated participants. You are the admin and a participant.")


with st.sidebar:
    st.header("Experiment controls")
    files = sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if "sim" in st.session_state:
        active_name = st.session_state.sim.card["experiment_id"]
        st.caption(f"Active session: {active_name}")
        if st.button("Set up another session"):
            del st.session_state.sim
            st.rerun()
        st.caption("This experiment uses the live LLM backend.")
    else:
        selected = st.selectbox("Saved session", ["New session"] + [p.name for p in files])
        if selected != "New session" and st.button("Load session"):
            try:
                saved = json.loads((RUNS / selected).read_text())
                settings = saved["card"].get("run_settings", {})
                if settings.get("backend", "live") != "live":
                    raise ValueError("This MVP runs live sessions only; this saved session uses another backend.")
                participant_model = settings.get("participant_model")
                research_model = settings.get("research_model")
                # Older runs stored OpenAI model IDs. Ignore those when loading under Anthropic.
                backend = AnthropicBackend(
                    model=participant_model if participant_model and participant_model.startswith("claude-") else None,
                    research_model=research_model if research_model and research_model.startswith("claude-") else None)
                st.session_state.sim = Simulation.load(RUNS / selected, backend)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load session: {type(exc).__name__}: {exc}")
        st.caption("Live mode uses ANTHROPIC_API_KEY from the environment or root .env. API usage may incur charges.")

if "sim" not in st.session_state:
    st.subheader("Set up a decision session")
    with st.form("new_session"):
        question = st.text_input("What decision is the group making?", "Where should our group go for a hiking trip?")
        human_name = st.text_input("Your participant name", "You")
        bot_count = st.slider("Number of simulated participants", 1, 5, 3)
        max_bot_turns = st.slider("Maximum bot messages in the session", 1, 60, 18)
        mapping_allowed = st.checkbox("Record your individual positions on the card", value=True)
        st.markdown("Give each simulated participant a role and high-level guidance. These are private to that bot.")
        profiles = []
        defaults = [
            ("Alex", "experienced hiker", "Prioritize challenging trails and explain what makes a route worthwhile."),
            ("Sam", "budget-conscious planner", "Keep costs manageable and ask for concrete cost assumptions."),
            ("Jordan", "accessibility-focused traveler", "Focus on practical access needs and travel burdens."),
            ("Casey", "curious explorer", "Value novelty but stay open to compromise."),
            ("Riley", "logistics-minded organizer", "Look for timing, availability, and coordination risks."),
        ]
        for i in range(bot_count):
            name, role, guidance = defaults[i]
            with st.expander(f"Simulated participant {i + 1}: {name}", expanded=i == 0):
                bname = st.text_input("Name", name, key=f"bot_name_{i}")
                brole = st.text_input("Role", role, key=f"bot_role_{i}")
                bguidance = st.text_area("High-level guidance", guidance, key=f"bot_guidance_{i}")
                bmap = st.checkbox("This simulated participant opts in to individual mapping", True, key=f"bot_map_{i}")
                bshare = st.checkbox("This participant allows positions on the shared card", True, key=f"bot_share_{i}")
                profiles.append({"name": bname, "role": brole, "guidance": bguidance,
                                 "mapping_allowed": bmap, "share_allowed": bshare})
        start = st.form_submit_button("Start experiment", type="primary")
    if start:
        try:
            backend = AnthropicBackend()
            sim = Simulation.create(question, human_name, profiles, max_bot_turns, backend,
                                    path=RUNS / "pending.json", mapping_allowed=mapping_allowed)
            session_path = RUNS / f"{sim.card['experiment_id']}.json"
            sim.path = session_path
            sim.save()
            (RUNS / "pending.json").unlink(missing_ok=True)
            st.session_state.sim = sim
            st.rerun()
        except Exception as exc:
            st.error(f"Could not start: {type(exc).__name__}: {exc}")
    st.stop()

sim = st.session_state.sim
card = sim.card
st.subheader(card["topic"]["title"])
st.info("Participant turns, card observation, requested claim checks, and Pact help all use the LLM.")

chat_col, card_col = st.columns([1.05, 1])
with chat_col:
    st.subheader("Conversation")
    for msg in card["messages"]:
        person = next(p for p in card["participants"] if p["id"] == msg["speaker_id"])
        with st.chat_message("user" if msg["source"] == "human" else "assistant"):
            st.caption(f"{person['name']} · {msg['id']}" + (" · simulated" if msg["source"] == "simulated" else ""))
            st.write(msg["text"])
    with st.form("human_message", clear_on_submit=True):
        human_text = st.text_area("Your message", max_chars=2500, key="human_message_text")
        sent = st.form_submit_button("Send to group")
    if sent:
        try:
            sim.add_human_message(human_text)
            st.rerun()
        except Exception as exc:
            st.error(f"Message not added: {exc}")
    next_col, round_col = st.columns(2)
    disabled = card["bot_turns"] >= card["max_bot_turns"]
    if next_col.button("Next participant", disabled=disabled):
        try:
            sim.bot_turn()
            st.rerun()
        except Exception as exc:
            st.error(f"Participant turn failed: {type(exc).__name__}: {exc}")
    if round_col.button("Next round", disabled=disabled):
        try:
            sim.next_round()
            st.rerun()
        except Exception as exc:
            st.error(f"Round stopped: {type(exc).__name__}: {exc}")
    st.caption(f"Bot messages: {card['bot_turns']}/{card['max_bot_turns']} · Model calls: {getattr(sim.backend, 'calls', 0)}")

with card_col:
    st.subheader("Decision card")
    st.caption(f"Card version {card['card_version']} · decisions require your explicit action")
    st.markdown("**Question**")
    st.write(card["topic"].get("problem") or card["topic"]["title"])
    for section, label in [("proposals", "Proposals"), ("criteria", "Criteria"), ("claims", "Claims"),
                           ("assumptions", "Assumptions"), ("arguments", "Arguments"),
                           ("positions", "Shared individual positions"), ("unresolved", "Open questions")]:
        entries = public_card(card).get(section, []) if section == "positions" else card.get(section, [])
        if entries:
            st.markdown(f"**{label}**")
            for entry in entries:
                st.write(f"{entry.get('title') or entry.get('statement') or entry.get('question') or json.dumps(entry)}")
                st.caption(f"{entry.get('id')} · source {', '.join(entry.get('source_message_ids', []))}")
    with st.expander("Explicit Pact actions", expanded=True):
        st.markdown("**Check a claim when asked**")
        if card["claims"]:
            claim_ids = [c["id"] for c in card["claims"]]
            choice = st.selectbox("Claim to check", claim_ids, format_func=lambda i: next(c["statement"] for c in card["claims"] if c["id"] == i))
            if st.button("Check selected claim"):
                with st.spinner("Checking the claim…"):
                    try:
                        sim.check_claim(choice)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Check failed: {type(exc).__name__}: {exc}")
        else:
            st.caption("A claim will appear here when it is identified in the conversation.")
        st.markdown("**Help when invited**")
        if st.button("Ask Pact for help"):
            with st.spinner("Pact is reviewing the card…"):
                try:
                    sim.ask_for_help()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pact could not help: {type(exc).__name__}: {exc}")
        for result in card["claim_checks"]:
            with st.expander(f"Claim check {result['id']} · {result['status']}"):
                st.write(result["finding"])
                for source in result.get("sources", []):
                    st.link_button(source["title"], source["url"])
        for help_item in card["facilitation"]:
            with st.expander(f"Pact help {help_item['id']}"):
                output = help_item["output"]
                if isinstance(output, dict):
                    st.markdown(f"**Where things stand:** {output.get('decision_state', '')}")
                    for key, label in [("factual_unknowns", "Factual unknowns"),
                                       ("preference_differences", "Preference differences"),
                                       ("tradeoffs", "Tradeoffs")]:
                        if output.get(key):
                            st.markdown(f"**{label}**")
                            for item in output[key]:
                                st.write(item)
                    st.markdown(f"**Next step:** {output.get('next_step', '')}")
                    if output.get("source_message_ids"):
                        st.caption("Sources: " + ", ".join(output["source_message_ids"]))
                else:
                    st.write(output)

    pending_items = [p for p in card["individual_mapping"]["pending_interpretations"]
                     if p["confirmation"]["status"] == "pending"]
    if pending_items:
        st.markdown("**Review your ambiguous interpretation**")
        own_pending = [p for p in pending_items if p["participant_id"] == "human_admin"]
        for item in own_pending:
            st.write(item["proposed_statement"])
            st.caption(f"Awaiting your confirmation · source {', '.join(item['source_message_ids'])}")
            response_text = st.text_input("Confirm or correct this interpretation", key=f"confirm_text_{item['id']}")
            confirm_col, correct_col = st.columns(2)
            if confirm_col.button("Confirm interpretation", key=f"confirm_{item['id']}"):
                try:
                    sim.confirm_interpretation(item["id"], "Confirm: " + response_text if response_text else "Confirm this interpretation.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Confirmation failed: {exc}")
            if correct_col.button("Submit correction", key=f"correct_{item['id']}"):
                try:
                    sim.confirm_interpretation(item["id"], response_text)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Correction failed: {exc}")
        other_pending = len(pending_items) - len(own_pending)
        if other_pending:
            st.caption(f"{other_pending} simulated-participant interpretation(s) remain pending and are excluded from confirmed card state.")

    if card["proposals"]:
        st.markdown("**Record a decision**")
        proposal_id = st.selectbox("Selected proposal", [p["id"] for p in card["proposals"]],
                                   format_func=lambda i: next(p.get("title", i) for p in card["proposals"] if p["id"] == i), key="decision_proposal")
        rationale = st.text_input("Rationale", key="decision_rationale")
        if st.button("Record decision snapshot"):
            try:
                sim.record_decision(proposal_id, rationale)
                st.rerun()
            except Exception as exc:
                st.error(f"Decision not recorded: {exc}")
    st.markdown("**Decision status**")
    st.json(card["decision"], expanded=False)
    st.download_button("Download session JSON", json.dumps({"card": card, "profiles": sim.profiles}, indent=2),
                       file_name=f"pact-{card['experiment_id']}.json", mime="application/json")

with st.expander("Engine audit and source details"):
    st.json({"change_log": card["change_log"], "audit": card["audit"],
             "pending_interpretations": card["individual_mapping"]["pending_interpretations"]}, expanded=False)
