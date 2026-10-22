"""Set QA — Streamlit UI. NUS CS3244 Kaggle Competition."""

import os, sys, io, csv, tempfile
import subprocess

def _is_streamlit():
    return "streamlit" in sys.modules

if not _is_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__]); sys.exit()

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from solution import parse_file, run, evaluate, process_story

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SetQA · NUS CS3244", page_icon="🎒", layout="wide")
G, R, B, Y, GR = "#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#95a5a6"

ETYPE_COLOR = {"pickup": G, "drop": R, "question": B, "move": GR, "none": GR,
               "transfer": Y, "negated_pickup": GR, "negated_drop": GR}
ETYPE_BADGE = {"pickup": "📦 PICKUP", "drop": "🗑️ DROP", "question": "❓ QUESTION",
               "move": "🚶 MOVE", "transfer": "🔄 TRANSFER",
               "negated_pickup": "🚫 NEG-PICKUP", "negated_drop": "🚫 NEG-DROP", "none": "·"}

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nus-3244-setqa")

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def upload_directory():
    return tempfile.TemporaryDirectory(prefix="ir-workbench-")

def load(uploaded, default):
    if uploaded is not None:
        import hashlib
        payload = uploaded.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        path = os.path.join(upload_directory().name, digest + ".txt")
        if not os.path.exists(path):
            with open(path, "wb") as handle:
                handle.write(payload)
        return path
    return default if os.path.isfile(default) else None

@st.cache_data(show_spinner=False)
def c_parse(p): return parse_file(p)

@st.cache_data(show_spinner=False)
def c_run(p):   return run(p)

@st.cache_data(show_spinner=False)
def c_eval(p):  return evaluate(c_run(p))

def inv_table(snap):
    if not snap: st.info("No characters yet."); return
    st.dataframe(pd.DataFrame([{"Character": k.title(), "Holding": ", ".join(v) or "—"}
                                for k, v in sorted(snap.items())]),
                 use_container_width=True, hide_index=True)

def color_badge(etype):
    return ETYPE_COLOR.get(etype, GR), ETYPE_BADGE.get(etype, etype.upper())

def event_card(ev, extra=""):
    c, b = color_badge(ev["event_type"])
    char_part = (f"<span style='color:{GR};font-size:.85em'> [{ev['character']}"
                 + (f" → <b style='color:{c}'>{ev['item']}</b>" if ev.get("item") else "")
                 + "]</span>") if ev.get("character") else ""
    st.markdown(
        f'<div style="border-left:4px solid {c};padding:5px 12px;margin:2px 0;border-radius:3px">'
        f'<span style="color:{GR};font-size:.78em">{ev["line_id"]}&nbsp;'
        f'<span style="color:{c}">{b}</span></span>&nbsp;&nbsp;'
        f'{ev["sentence"]}{char_part}{extra}</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎒 SetQA")
    st.caption("NUS CS3244 · Kaggle")
    st.divider()
    up_train = st.file_uploader("train.txt", type="txt", key="tr")
    up_test  = st.file_uploader("test.txt",  type="txt", key="te")
    train_path = load(up_train, os.path.join(DEFAULT, "train.txt"))
    test_path  = load(up_test,  os.path.join(DEFAULT, "test.txt"))
    st.divider()
    st.success("✅ train.txt") if train_path else st.warning("⚠️ train.txt missing")
    st.success("✅ test.txt")  if test_path  else st.warning("⚠️ test.txt missing")

fingerprint = (train_path, test_path)
if st.session_state.get("input_fingerprint") != fingerprint:
    for key in ("test_results", "summary"):
        st.session_state.pop(key, None)
    st.session_state["input_fingerprint"] = fingerprint

tab1, tab2, tab3, tab4 = st.tabs(["📊 Explorer", "🔍 Walkthrough", "📈 Evaluation", "📤 Submit"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Explorer")
    if not train_path: st.error("Upload train.txt to continue.")
    else:

        with st.spinner("Parsing…"):
            stories = c_parse(train_path)

        sents     = [s for st_ in stories for s in st_["sentences"]]
        questions = [s for s in sents if s["is_question"]]
        chars     = set()
        item_ctr  = Counter()

        for story in stories:
            for r in process_story(story):
                chars.add(r["character"])
        for q in questions:
            if q["answer"] and q["answer"] != "-1":
                for w in q["answer"].split():
                    if not w.lstrip("-").isdigit(): item_ctr[w.lower()] += 1

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📖 Stories",    len(stories))
        c2.metric("❓ Questions",  len(questions))
        c3.metric("👤 Characters", len(chars))
        c4.metric("🎒 Items",      len(item_ctr))
        st.divider()

        # Story browser
        st.subheader("Story Browser")
        opts   = {f"Story {s['story_id']} ({len(s['sentences'])} lines)": s for s in stories}
        sel    = st.selectbox("Select story", list(opts.keys()))
        story  = opts[sel]
        for s in story["sentences"]:
            if s["is_question"]:
                ans = s["answer"] or "?"
                st.markdown(
                    f'<div style="background:#1a3a5c;border-left:4px solid {B};padding:6px 12px;'
                    f'margin:2px 0;border-radius:4px"><span style="color:{GR};font-size:.8em">'
                    f'{s["line_id"]}</span>&nbsp;&nbsp;<span style="color:{B}">{s["text"]}</span>'
                    f'&nbsp;&nbsp;<span style="background:#144a2a;color:{G};padding:2px 8px;'
                    f'border-radius:12px;font-size:.85em">→ {ans}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:4px 12px;margin:2px 0">'
                            f'<span style="color:{GR};font-size:.8em">{s["line_id"]}</span>'
                            f'&nbsp;&nbsp;{s["text"]}</div>', unsafe_allow_html=True)
        st.divider()

        # Analytics
        st.subheader("Analytics")
        n_nothing = sum(1 for q in questions if q["answer"] == "-1")
        n_multi   = sum(1 for q in questions if q["answer"] and q["answer"] != "-1" and " " in q["answer"])
        n_single  = len(questions) - n_nothing - n_multi

        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(px.pie(values=[n_nothing, n_single, n_multi],
                names=["Nothing (-1)", "Single", "Multi"],
                color_discrete_sequence=[GR, G, B], title="Answer Type Distribution"),
                use_container_width=True)
        with cr:
            q_per = [sum(1 for s in st_["sentences"] if s["is_question"]) for st_ in stories]
            st.plotly_chart(px.histogram(x=q_per, nbins=20, title="Questions per Story",
                labels={"x":"#Questions","y":"Stories"}, color_discrete_sequence=[B]),
                use_container_width=True)

        cl2, cr2 = st.columns(2)
        with cl2:
            if item_ctr:
                top = item_ctr.most_common(15)
                st.plotly_chart(px.bar(x=[i[1] for i in top], y=[i[0] for i in top],
                    orientation="h", title="Top 15 Items",
                    labels={"x":"Count","y":"Item"}, color_discrete_sequence=[G]).update_layout(
                    yaxis={"autorange":"reversed"}), use_container_width=True)
        with cr2:
            char_ctr = Counter()
            for st_ in stories:
                for r in process_story(st_): char_ctr[r["character"]] += 1
            top_c = char_ctr.most_common(10)
            if top_c:
                st.plotly_chart(px.bar(x=[c[1] for c in top_c], y=[c[0] for c in top_c],
                    orientation="h", title="Top 10 Characters",
                    labels={"x":"Questions","y":"Character"}, color_discrete_sequence=[Y]).update_layout(
                    yaxis={"autorange":"reversed"}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Walkthrough")
    if not train_path: st.error("Upload train.txt first.")
    else:

        with st.spinner("Parsing…"):
            wt_stories = c_parse(train_path)

        sel_story = st.selectbox("Select story",
            {f"Story {s['story_id']}": s for s in wt_stories}, key="wt")
        wt = {f"Story {s['story_id']}": s for s in wt_stories}[sel_story]
        mode = st.radio("Mode", ["Full Story", "Step Through"], horizontal=True)
        wt_results = process_story(wt)
        all_events = wt_results[-1]["events"] if wt_results else []

        if mode == "Step Through":
            if not all_events: st.info("No events."); st.stop()
            step = st.slider("Step", 1, len(all_events), 1)
            ev   = all_events[step - 1]
            c, b = color_badge(ev["event_type"])
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:6px solid {c};padding:12px 20px;'
                f'border-radius:6px;margin-bottom:12px"><div style="color:{GR};font-size:.8em">'
                f'Line {ev["line_id"]} — <span style="color:{c}">{b}</span></div>'
                f'<div style="font-size:1.1em;margin-top:4px">{ev["sentence"]}</div>'
                + (f'<div style="margin-top:6px">Character: <b>{ev["character"]}</b>'
                   + (f' → <b style="color:{c}">{ev["item"]}</b>' if ev.get("item") else "")
                   + "</div>" if ev.get("character") else "")
                + "</div>", unsafe_allow_html=True)
            if ev["event_type"] == "question":
                for r in wt_results:
                    if r["events"] and r["events"][-1]["line_id"] == ev["line_id"]:
                        ca, cb = st.columns(2)
                        ca.metric("Predicted", r["predicted_answer"])
                        if r["expected_answer"] is not None:
                            cb.metric("Expected", f"{r['expected_answer']} {'✅' if r['is_correct'] else '❌'}")
                        break
            st.subheader("📦 Inventory at this step")
            inv_table(ev["inventory_snapshot"])
        else:
            for ev in all_events: event_card(ev)
            st.divider()
            st.subheader("📦 Final Inventory")
            if all_events: inv_table(all_events[-1]["inventory_snapshot"])
            if wt_results:
                st.subheader("❓ Questions")
                st.dataframe(pd.DataFrame([{
                    "Q#": r["question_id"], "Question": r["question_text"],
                    "Holding": ", ".join(r["inventory_at_q"]) or "nothing",
                    "Predicted": r["predicted_answer"], "Expected": r["expected_answer"] or "—",
                    "✓": "✅" if r["is_correct"] else ("❌" if r["is_correct"] is False else "—"),
                } for r in wt_results]), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Evaluation")
    if not train_path: st.error("Upload train.txt first.")
    else:

        if st.button("🔍 Run Evaluation", type="primary"):
            with st.spinner("Evaluating…"):
                st.session_state["summary"] = c_eval(train_path)

        if "summary" in st.session_state:
            s   = st.session_state["summary"]
            acc = s["accuracy"]

            acc_color = G if acc >= .90 else Y if acc >= .75 else R
            st.markdown(
                f'<div style="background:#1a1a2e;padding:24px;border-radius:12px;border:2px solid '
                f'{acc_color};text-align:center;margin-bottom:20px">'
                f'<div style="font-size:3em;font-weight:bold;color:{acc_color}">{acc:.2%}</div>'
                f'<div style="color:{GR}">Accuracy</div></div>', unsafe_allow_html=True)

            c1,c2,c3 = st.columns(3)
            c1.metric("Total",   s["total"])
            c2.metric("Correct", s["correct"], delta=f"+{s['correct']}")
            c3.metric("Wrong",   s["wrong"],   delta=f"-{s['wrong']}", delta_color="inverse")
            st.divider()

            st.subheader("By Answer Type")
            st.dataframe(pd.DataFrame([{"Type": n.title(), "Total": v["total"],
                "Correct": v["correct"], "Accuracy": f"{v['accuracy']:.2%}"}
                for n, v in s["by_answer_type"].items() if v["total"]]),
                use_container_width=True, hide_index=True)
            st.divider()

            errors = s.get("error_details", [])
            if errors:
                from collections import Counter as C
                etype_counts = C(e["error_type"] for e in errors)
                st.subheader("Error Breakdown")
                st.plotly_chart(px.bar(x=list(etype_counts.keys()), y=list(etype_counts.values()),
                    labels={"x":"Error Type","y":"Count"},
                    color=list(etype_counts.keys()),
                    color_discrete_sequence=[R,Y,B,GR,G], title="Error Types"),
                    use_container_width=True)
                st.divider()

                st.subheader("Error Details")
                df = pd.DataFrame([{"Story": e["story_id"], "Q#": e["question_id"],
                    "Character": e["character"], "Question": e["question_text"],
                    "Expected": e["expected"], "Predicted": e["predicted"],
                    "Error Type": e["error_type"]} for e in errors])

                sel_types = st.multiselect("Filter by error type",
                    options=sorted(df["Error Type"].unique()),
                    default=sorted(df["Error Type"].unique()))
                df_f = df[df["Error Type"].isin(sel_types)] if sel_types else df
                st.caption(f"Showing {len(df_f)} / {len(errors)} errors")
                st.dataframe(df_f, use_container_width=True, height=420, hide_index=True,
                    column_config={"Question": st.column_config.TextColumn(width="large"),
                                   "Error Type": st.column_config.TextColumn(width="small")})
                st.divider()

                buf = io.StringIO()
                w   = csv.writer(buf)
                w.writerow(["Story","Q#","Character","Question","Expected","Predicted","Error Type","Expected Offsets","Inventory"])
                for e in errors:
                    inv = ", ".join(f"{k}:{','.join(v)}" for k,v in e["inventory_snapshot"].items())
                    w.writerow([e["story_id"],e["question_id"],e["character"],e["question_text"],
                                e["expected"],e["predicted"],e["error_type"],e["expected_offsets"],inv])
                st.download_button("⬇️ Download errors.csv", buf.getvalue(),
                                   "errors.csv", "text/csv", type="primary")
            else:
                st.success("🎉 Zero errors on training set!")
        else:
            st.info("Click **Run Evaluation** to start.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GENERATE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Generate Submission")
    if not test_path: st.error("Upload test.txt to continue.")
    else:

        if st.button("🚀 Run on Test Set", type="primary"):
            prog = st.progress(0, text="Parsing test.txt…")
            test_stories = parse_file(test_path)
            results = []
            for i, s in enumerate(test_stories):
                results.extend(process_story(s))
                prog.progress((i+1)/len(test_stories), text=f"Story {i+1}/{len(test_stories)}…")
            prog.progress(1.0, text="Done!")
            st.session_state["test_results"] = results

        if "test_results" in st.session_state:
            res = st.session_state["test_results"]
            n_nothing = sum(1 for r in res if r["predicted_answer"] == "-1")
            n_multi   = sum(1 for r in res if " " in r["predicted_answer"])
            n_single  = len(res) - n_nothing - n_multi

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total",       len(res))
            c2.metric("Nothing (-1)", n_nothing)
            c3.metric("Single",      n_single)
            c4.metric("Multi",       n_multi)

            st.subheader("Preview (first 20)")
            st.dataframe(pd.DataFrame([{
                "textID": f"{r['story_id']}_{r['question_id']}",
                "Answer": r["predicted_answer"],
                "Type": "nothing" if r["predicted_answer"]=="-1" else ("multi" if " " in r["predicted_answer"] else "single")
            } for r in res[:20]]), use_container_width=True, hide_index=True)

            ids = [f"{r['story_id']}_{r['question_id']}" for r in res]
            if len(ids) == len(set(ids)): st.success(f"✅ {len(ids)} unique IDs — no duplicates.")
            else: st.error("❌ Duplicate IDs detected!")

            buf = io.StringIO()
            buf.write("textID,sortedAnswerList\n")
            for r in res: buf.write(f"{r['story_id']}_{r['question_id']},{r['predicted_answer']}\n")
            st.download_button("⬇️ Download submission.csv", buf.getvalue().encode(),
                               "submission.csv", "text/csv", type="primary")
