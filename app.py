"""
Logistics Management (LM) Course Tutor  -  Streamlit + Groq (Llama)
Reads the course files in ./course_files, retrieves the relevant excerpts for each question,
and answers in the voice of a Professor of Supply Chain, Operations Research and Cost Accounting.
Run:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

import knowledge as K

COURSE_DIR = Path(__file__).parent / "course_files"
PRESET_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
CONTEXT_SIZES = {"Small (fast, fits tight free limits)": 5000, "Medium (recommended)": 8000, "Large (more detail)": 12000}

STARTERS = {
    "ParcelXpress: phantom freight vs absorption": "Explain phantom freight and freight absorption in the ParcelXpress case, with worked numbers from the files.",
    "Warehouse space: walk through the Solver model": "Reverse-engineer the Solver model for the warehouse space allocation case: objective, decision variables, constraints and algorithm.",
    "3PL selection: how are providers scored?": "Explain how the Third-Party Logistics providers are scored and which one should be chosen, step by step.",
    "Blinkit: sequential vs integrated model": "Explain the difference between the sequential and the integrated Blinkit dark store models and what the capacity and rider-loss tests teach.",
    "Rail or road: which mode and why?": "Explain the rail versus road decision logic for Vidarbha Agro Foods and the key cost drivers.",
    "Transportation model: read the Solver": "Explain the Transport Optimization Solver model in plain English, including which constraints are binding.",
}

SYSTEM_PROMPT = """You are an expert Professor of Supply Chain and Logistics Management, Operations Research, and Managerial Cost Accounting, acting as an interactive teaching assistant for one course. Students ask about the case studies, Excel Solver workbooks and lecture slides of this course.

GROUNDING
- You receive COURSE EXCERPTS retrieved from the course files. Base your answer on them and cite the source after the claim, like (ParcelXpress_solution.xlsx > sheet 'Q2 Phantom Costs' row 15-19).
- If the excerpts do not contain what is needed, say so plainly, then give general textbook knowledge clearly labelled "General knowledge, not from the course files".
- Never invent numbers. Quote figures only from the excerpts or compute them step by step from numbers in the excerpts.

STYLE RULES
1. Strict abbreviation policy: never use an abbreviation or acronym before its full term. Write the full term first with the abbreviation in parentheses on first mention, for example Less Than Truckload (LTL), Third-Party Logistics (3PL), Linear Programming (LP), Economic Order Quantity (EOQ), Reorder Point (ROP), Distribution Center (DC). Afterwards the abbreviation may be used.
2. Descriptive mathematics: never show a bare formula. Say in full words what it means before or alongside any notation. Example: "Total Revenue Contribution under a Flat Rate policy equals the Flat Rate fee minus the Direct Variable Shipping Cost for that zone, multiplied by the Daily Demand Volume for that zone."
3. Explain thoroughly in descriptive prose. Translate constraints, shadow prices and cost allocations into managerial meaning.

DOMAIN KNOWLEDGE
- Phantom Freight: a customer near the distribution center pays a uniform flat rate above the seller's actual direct shipping cost; the seller keeps the difference as phantom margin (flat rate minus actual cost, times shipment volume).
- Freight Absorption: a customer in a distant, high-cost zone pays a flat rate below the actual carrier cost; the seller absorbs the loss out of product gross margin (actual cost minus flat rate, times shipment volume).
- For any Solver model use four pillars: (a) Objective Function: maximize profit, maximize service or minimize cost, and the business meaning of the target cell; (b) Decision Variables: the physical actions the manager controls; (c) Operational Constraints: the real bottlenecks the formulas represent, and which are binding; (d) Algorithm: Linear Programming (LP) with Simplex, integer or non-linear method, why it was chosen, and how shadow prices (dual values) show the marginal value of relaxing a bottleneck. A saved Solver model does not store the sensitivity report, so only describe shadow prices conceptually or from binding constraints.
- Frameworks: network design (direct versus hub-and-spoke), Economic Order Quantity (EOQ), Reorder Point (ROP), cycle versus safety stock, risk pooling, and transport mode trade-offs against pipeline holding cost.

ANSWER FLOW
1. Name the case or tab. 2. State the managerial decision and the financial trade-off. 3. Walk through the mechanics step by step with worked numbers from the files. 4. End with a managerial recommendation: strategic risk, operational feasibility and a decision rule."""


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def get_secret(name):
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val or os.environ.get(name)


@st.cache_resource(show_spinner="Reading the course files...")
def load_course_kb():
    return K.build_kb_from_dir(str(COURSE_DIR))


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return Groq(api_key=api_key)


def build_messages(question, history, excerpts):
    """System persona + last few turns + the new question carrying its retrieved excerpts."""
    ctx = "\n\n".join(f"[{i}] SOURCE: {c.source} > {c.loc}\n{c.text}" for i, c in enumerate(excerpts, 1)) or "(no matching excerpts found)"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": f"COURSE EXCERPTS:\n{ctx}\n\nQUESTION: {question}"})
    return msgs


def stream_llm(client, model, messages):
    stream = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=1300, stream=True)
    for ev in stream:
        if ev.choices and ev.choices[0].delta.content:
            yield ev.choices[0].delta.content


def is_size_error(e):
    t = str(e).lower()
    return any(x in t for x in ("413", "too large", "rate_limit", "tokens per minute", "request too large"))


def ask_professor(client, model, question, kb, ctx_chars, top_k, history=(), extra_context=""):
    """Retrieve -> prompt -> stream. Retries once with half the context if a free-tier size limit is hit."""
    last_err = None
    for size in (ctx_chars, ctx_chars // 2):
        excerpts = kb.search(question, k=top_k, max_chars=size)
        q = question + (f"\n\nADDITIONAL MODEL DESCRIPTION:\n{extra_context}" if extra_context else "")
        try:
            text = st.write_stream(stream_llm(client, model, build_messages(q, list(history), excerpts)))
            return text, excerpts
        except Exception as e:
            last_err = e
            if not is_size_error(e):
                break
            st.info("The request was too large for the free tier, retrying with fewer excerpts...")
    raise last_err


def show_sources(excerpts):
    if excerpts:
        with st.expander(f"📎 Sources used ({len(excerpts)})"):
            for c in excerpts:
                st.markdown(f"**{c.source}** > {c.loc}")
                st.caption(c.text[:450] + ("..." if len(c.text) > 450 else ""))


def explain_error(e):
    st.error("The Groq call failed. Check that your API key is valid, you are within the free quota, and the model ID exists "
             f"(console.groq.com/docs/models).\n\nDetails: {e}")


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def tutor_tab(kb, client, model, ctx_chars, top_k):
    st.session_state.setdefault("chat", [])
    st.markdown("**Start with a case, or type your own question below.**")
    cols = st.columns(3)
    for i, (label, q) in enumerate(STARTERS.items()):
        if cols[i % 3].button(label, key=f"starter{i}"):
            st.session_state.pending = q
    if st.button("🧹 Clear conversation"):
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    typed = st.chat_input("Ask about any case, Solver model, formula or lecture concept...")
    question = typed or st.session_state.pop("pending", None)
    if not question:
        return
    # short follow-ups borrow the previous question so retrieval stays on topic
    prev = next((m["content"] for m in reversed(st.session_state.chat) if m["role"] == "user"), "")
    retrieval_q = question if len(question.split()) > 7 or not prev else prev + " " + question

    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if client is None:
            st.warning("No GROQ_API_KEY found, so here are the most relevant course excerpts (library-only mode).")
            excerpts = kb.search(retrieval_q, k=top_k, max_chars=ctx_chars)
            answer = "\n\n".join(f"**{c.source}** > {c.loc}\n\n{c.text}" for c in excerpts) or "No matching excerpts found."
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
            return
        try:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[:-1]]
            text, excerpts = ask_professor(client, model, retrieval_q if retrieval_q != question else question, kb,
                                           ctx_chars, top_k, history)
            show_sources(excerpts)
            st.session_state.chat.append({"role": "assistant", "content": text, "sources": excerpts})
        except Exception as e:
            explain_error(e)
            st.session_state.chat.pop()


def xray_tab(kb, client, model, ctx_chars, top_k):
    st.markdown("Every workbook that still has its **Excel Solver settings saved** is decoded below into the four pillars: "
                "objective, decision variables, constraints and algorithm. Constraints show whether they are **binding** "
                "(no slack left), which is where a shadow price would be non-zero.")
    books = sorted(n for n in kb.files if n.lower().endswith(".xlsx"))
    name = st.selectbox("Workbook", books)
    models = K.extract_solver_models(kb.files[name])
    if not models:
        st.info("No Solver model is saved in this workbook. It is a formula-driven model, so ask about it in the Tutor tab.")
        return
    for m in models:
        st.subheader(f"Sheet '{m['sheet']}'")
        a, b, c = st.columns(3)
        a.metric("Goal", m["sense"])
        b.metric("Algorithm", m["engine"])
        c.metric("Objective value", f"{m['objective']['value']:,.2f}" if m["objective"] and isinstance(m["objective"]["value"], (int, float)) else "n/a")
        if m["objective"]:
            st.markdown(f"**Objective (target cell `{m['objective']['ref']}`)**: {m['objective']['label'] or 'unlabelled cell'}")
        if m["changing"]:
            ch = m["changing"]
            st.markdown(f"**Decision variables (changing cells `{ch['ref']}`, {ch['n']} cells)**: {', '.join(ch['labels']) or 'n/a'}")
        rows = [{"#": i, "Left side": f"{k['lhs']} ({k['lhs_label'] or 'cells'})", "Relation": k["rel"], "Right side / limit": k["rhs"],
                 "Cells": k["n"], "Binding cells": k["binding"] if k["binding"] is not None else "-",
                 "Smallest slack": round(k["min_slack"], 4) if k["min_slack"] is not None else "-"}
                for i, k in enumerate(m["constraints"], 1)]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        if st.button("🎓 Explain this model like a professor", key=f"xr_{name}_{m['sheet']}"):
            if client is None:
                st.warning("Add a GROQ_API_KEY to get the professor's explanation.")
            else:
                q = f"Reverse-engineer the Solver model on sheet '{m['sheet']}' of {name} using the four pillars and give managerial recommendations."
                try:
                    with st.chat_message("assistant"):
                        _, ex = ask_professor(client, model, q, kb, ctx_chars, top_k, extra_context=K.solver_report_text(m))
                        show_sources(ex)
                except Exception as e:
                    explain_error(e)


def library_tab(kb):
    st.markdown("Everything the tutor can read. Search works even without an API key.")
    st.dataframe(pd.DataFrame(kb.summary()), hide_index=True)
    q = st.text_input("Search the course material")
    if q:
        for c in kb.search(q, k=8, max_chars=20000):
            with st.expander(f"{c.source} > {c.loc}"):
                st.text(c.text)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="LM Course Tutor", page_icon="📦", layout="wide")
    st.title("📦 Logistics Management Course Tutor")
    st.caption("Ask about the cases, Solver workbooks and Session 1-8 slides. Answers cite the course files.")

    kb = load_course_kb()
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = get_secret("GROQ_API_KEY")
        st.success("Groq API key detected") if api_key else st.warning("No GROQ_API_KEY: library-only mode")
        default_model = get_secret("GROQ_MODEL")
        preset = st.selectbox("Groq model", PRESET_MODELS, index=PRESET_MODELS.index(default_model) if default_model in PRESET_MODELS else 0)
        custom = st.text_input("Custom model ID (optional)", help="Model names change; see console.groq.com/docs/models")
        model = custom.strip() or preset
        ctx_label = st.selectbox("Context size", list(CONTEXT_SIZES), index=1)
        top_k = st.slider("Excerpts per question", 3, 10, 6)

        st.header("📤 Add your own files")
        ups = st.file_uploader("PDF, DOCX, XLSX, TXT", type=["pdf", "docx", "xlsx", "txt", "md", "csv"], accept_multiple_files=True)
        if ups:
            sig = tuple((u.name, u.size) for u in ups)
            if st.session_state.get("up_sig") != sig:
                st.session_state.up_kb = K.build_kb_from_uploads([(u.name, u.getvalue()) for u in ups])
                st.session_state.up_sig = sig
            kb = kb.merged_with(st.session_state.up_kb)
        st.caption(f"{len(kb.files)} files, {kb.N} searchable chunks")

    client = get_client(api_key) if api_key else None
    t1, t2, t3 = st.tabs(["💬 Tutor", "🔬 Solver X-Ray", "📚 Library"])
    with t1:
        tutor_tab(kb, client, model, CONTEXT_SIZES[ctx_label], top_k)
    with t2:
        xray_tab(kb, client, model, CONTEXT_SIZES[ctx_label], top_k)
    with t3:
        library_tab(kb)


if __name__ == "__main__":
    main()
