"""All dashboard pages (Streamlit + Plotly)."""
import json, os, subprocess, sys
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.config import ARTIFACTS, DATA_PATH, ROOT
from src.preprocessing import CAT, ENG_FEATURES, RAW_FEATURES, add_engineered_features
from src.reasoning import fol_rule, forward_chain, proof_lines, prove

CSS = """<style>
.kpi{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 18px;background:rgba(128,128,128,.07);height:100%}
.kpi .l{opacity:.7;font-size:.85rem}.kpi .v{font-size:1.9rem;font-weight:700;line-height:1.25}.kpi .s{opacity:.6;font-size:.78rem}
.step{border-left:4px solid #4C6FFF;padding:6px 12px;background:rgba(76,111,255,.08);border-radius:6px;font-size:.88rem;height:100%}
</style>"""
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
C = {"Logistic Regression": "#4C6FFF", "Decision Tree": "#22B07D", "Neural Network (MLP)": "#B04CFF", "Random Forest": "#00B8D9", "Rule-based (forward chaining)": "#F5A524"}
YES, NO = "#22B07D", "#EF553B"


@st.cache_resource(show_spinner="First start: training the models (about a minute)...")
def load():
    if not (ARTIFACTS / "curves.json").exists():
        subprocess.check_call([sys.executable, "-m", "src.train"], cwd=ROOT)
    rd = lambda f: json.load(open(ARTIFACTS / f"{f}.json"))
    return joblib.load(ARTIFACTS / "models.joblib"), rd("rules"), rd("metrics"), rd("meta"), rd("curves")


@st.cache_data
def data():
    d = pd.read_csv(DATA_PATH, sep=";"); d["subscribed"] = (d["y"] == "yes").astype(int); return d


models, kb, metrics, meta, curves = load()
rules, cats, D = kb["rules"], meta["cats"], meta["defaults"]
NAMES = list(metrics); ML = NAMES[:-1]; RULE = NAMES[-1]; KS = [n.split()[0] for n in ML] + ["Rules"]   # ML learners, rule engine last
BEST = max(ML, key=lambda n: metrics[n]["f1"])


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def show(fig, h=330, **kw):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=46, b=8), **kw); st.plotly_chart(fig, use_container_width=True)


def rate_bar(g, x, base, title):
    fig = px.bar(g, x=x, y="rate", text=g["rate"].map("{:.0%}".format), color="rate", color_continuous_scale="Blues", title=title)
    fig.add_hline(y=base, line_dash="dash", annotation_text=f"average {base:.1%}"); fig.update_coloraxes(showscale=False)
    fig.update_yaxes(tickformat=".0%", title="Subscription rate"); fig.update_traces(textposition="outside"); return fig


# =============================================================== Overview
def overview():
    df = data(); N = len(df); base = df.subscribed.mean(); bm = metrics[BEST]
    st.title("🏦 Term-Deposit Subscription Dashboard")
    st.caption("Machine learning + first-order-logic reasoning on the UCI Bank Marketing campaign data.")
    c = st.columns(4)
    kpi(c[0], "Clients analysed", f"{N:,}", f"{df.shape[1]-2} client & campaign attributes")
    kpi(c[1], "Subscription rate", f"{base:.1%}", f"{df.subscribed.sum():,} clients said yes")
    kpi(c[2], "Best model F1", f"{bm['f1']:.3f}", f"{BEST} · recall {bm['rec']:.0%}")
    kpi(c[3], "ROC-AUC", f"{bm['roc']:.3f}", "0.5 = random guessing, 1.0 = perfect")
    st.write("")
    l, r = st.columns([1, 1.5])
    with l:
        fig = go.Figure(go.Pie(labels=["no", "yes"], values=[(df.subscribed == 0).sum(), df.subscribed.sum()], hole=.62, marker_colors=[NO, YES], sort=False))
        fig.update_layout(title="Target distribution (y)", annotations=[dict(text=f"{base:.1%}<br>yes", showarrow=False, font_size=20)]); show(fig, 340)
    with r:
        g = df.groupby("month").subscribed.mean().reindex(MONTHS).dropna().rename("rate").reset_index()
        show(rate_bar(g, "month", base, "Subscription rate by month of last contact"), 340)
    pr = df.groupby("poutcome").subscribed.mean(); mo = df.groupby("month").subscribed.mean(); jb = df.groupby("job").subscribed.mean()
    hs = df.groupby("housing").subscribed.mean()
    st.subheader("Key insights")
    a, b = st.columns(2)
    a.info(f"**Previous success is the strongest signal:** {pr['success']:.0%} of clients with a successful earlier campaign subscribed, versus {base:.1%} overall.")
    a.info(f"**Best month:** {mo.idxmax()} ({mo.max():.0%}). **Best job group:** {jb.idxmax()} ({jb.max():.0%}).")
    b.info(f"**Debt lowers the odds:** {hs['yes']:.1%} of clients with a housing loan subscribed versus {hs['no']:.1%} without.")
    b.warning(f"**Imbalanced target:** always answering \"no\" is {1-base:.1%} accurate, so precision, recall and F1 are used instead of accuracy.")
    st.subheader("How the project works")
    s = st.columns(5)
    for col, (t, d) in zip(s, [("1 · Clean", "Remove duplicates, fix `pdays = -1`, keep 'unknown' as a category"), ("2 · Encode & scale", "One-hot + binary encoding, clip outliers, standardise (train only)"),
                               ("3 · Split", "Stratified 80:20 train/test; test set untouched until the end"), ("4 · Model", "Neural network (MLP), Logistic Regression, Decision Tree, Random Forest and a rule base with forward/backward chaining"),
                               ("5 · Evaluate", "Precision, recall, F1, ROC-AUC, PR-AUC, gains chart and plain-language inference")]):
        col.markdown(f"<div class='step'><b>{t}</b><br>{d}</div>", unsafe_allow_html=True)


# =============================================================== Explorer
def explorer():
    df = data(); base = df.subscribed.mean()
    st.title("📈 Data Explorer"); st.caption("Descriptive view of the full dataset (45,211 clients).")
    t = st.tabs(["Feature vs outcome", "Distributions & outliers", "Correlation", "Data quality"])
    with t[0]:
        f = st.selectbox("Choose a feature", RAW_FEATURES + ["duration"], index=RAW_FEATURES.index("poutcome"))
        if f == "duration": st.warning("Duration shows a very strong link with the outcome, but it is only known after the call, so it is **not** used by the models (data leakage).")
        if pd.api.types.is_numeric_dtype(df[f]):
            g = df.assign(bin=pd.qcut(df[f], 10, duplicates="drop")).groupby("bin", observed=True).agg(rate=("subscribed", "mean"), n=("subscribed", "size")).reset_index()
            g["bin"] = g["bin"].astype(str); x = "bin"
        else:
            g = df.groupby(f).agg(rate=("subscribed", "mean"), n=("subscribed", "size")).reset_index(); x = f
            g = g.set_index(f).reindex(MONTHS).dropna().reset_index() if f == "month" else g.sort_values("rate", ascending=False)
        a, b = st.columns(2)
        with a: show(rate_bar(g, x, base, f"Subscription rate by {f}"), 380)
        with b: show(px.bar(g, x=x, y="n", title=f"Number of clients by {f}", color_discrete_sequence=["#8FA3FF"]), 380)
    with t[1]:
        num = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        f = st.selectbox("Numeric feature", num, key="distf"); trim = st.checkbox("Hide the most extreme 0.5% of values", value=True)
        d = df[df[f] <= df[f].quantile(.995)] if trim else df
        show(px.histogram(d, x=f, color="y", barmode="overlay", nbins=50, opacity=.65, marginal="box", color_discrete_map={"no": NO, "yes": YES}, title=f"Distribution of {f} by outcome"), 440)
        q1, q3 = df[f].quantile([.25, .75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n = int(((df[f] < lo) | (df[f] > hi)).sum()); st.caption(f"IQR rule: {n:,} values ({n/len(df):.1%}) lie outside [{lo:.0f}, {hi:.0f}]. They are valid clients, so they are clipped rather than removed.")
    with t[2]:
        cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous", "subscribed"]
        fig = px.imshow(df[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation of numeric features with the target")
        show(fig, 500)
    with t[3]:
        u = {c: (df[c] == "unknown").mean() for c in df.select_dtypes(exclude="number") if c != "y"}; u["pdays (-1 = never contacted)"] = (df["pdays"] == -1).mean()
        u = pd.Series(u); u = u[u > 0].sort_values()
        show(px.bar(x=u.values, y=u.index, orientation="h", text=[f"{v:.1%}" for v in u.values], title="Hidden missing values ('unknown' or placeholder)", color_discrete_sequence=["#F5A524"]).update_xaxes(tickformat=".0%", title="Share of rows"), 360)
        st.caption(f"True blank cells: {int(df.isna().sum().sum())}. 'unknown' is kept as its own category.")


# =============================================================== Performance
def performance():
    st.title("📊 Model Performance"); st.caption(f"Measured once on {meta['n_test']:,} unseen test clients. Positive class = subscribed.")
    mt = pd.DataFrame(metrics).T.rename(columns=dict(acc="Accuracy", prec="Precision", rec="Recall", f1="F1", roc="ROC-AUC", pr="PR-AUC"))
    show_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    a, b = st.columns([1.1, 1])
    with a:
        st.dataframe(mt[show_cols].astype(float).style.format("{:.3f}", na_rep="—"), use_container_width=True)
        st.caption("Always answering 'no' scores {:.1%} accuracy, so accuracy alone is misleading.".format(1 - meta["pos_rate_test"]))
    with b:
        long = mt[["Precision", "Recall", "F1"]].astype(float).reset_index().melt("index")
        show(px.bar(long, x="variable", y="value", color="index", barmode="group", color_discrete_map=C, title="Precision / Recall / F1").update_layout(legend_title=None, legend=dict(orientation="h", y=-.2)), 330)
    a, b = st.columns(2)
    fig = go.Figure(); fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random")
    for n in ML: fig.add_scatter(x=curves[n]["roc"][0], y=curves[n]["roc"][1], mode="lines", name=f"{n} (AUC {metrics[n]['roc']:.3f})", line=dict(color=C[n]))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(orientation="h", y=-.25)); a.plotly_chart(fig.update_layout(height=380, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    fig = go.Figure(); fig.add_hline(y=meta["pos_rate_test"], line_dash="dash", annotation_text="no-skill")
    for n in ML: fig.add_scatter(x=curves[n]["pr"][0], y=curves[n]["pr"][1], mode="lines", name=f"{n} (PR-AUC {metrics[n]['pr']:.3f})", line=dict(color=C[n]))
    fig.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision", legend=dict(orientation="h", y=-.25)); b.plotly_chart(fig.update_layout(height=380, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    a, b = st.columns(2)
    fig = go.Figure(); fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random calling")
    for n in ML: fig.add_scatter(x=curves[n]["gain"][0], y=curves[n]["gain"][1], mode="lines", name=n, line=dict(color=C[n]))
    (tn, fp), (fn, tp) = metrics[RULE]["cm"]; N = tn + fp + fn + tp
    fig.add_scatter(x=[(tp + fp) / N], y=[tp / (tp + fn)], mode="markers", marker=dict(size=12, color=C[RULE]), name="Rules")
    fig.update_layout(title="Cumulative gains: share of subscribers reached vs share of clients called", xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="Clients called (best-scored first)", yaxis_title="Subscribers reached", legend=dict(orientation="h", y=-.25))
    a.plotly_chart(fig.update_layout(height=400, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    cv = curves[BEST]; ctr = [(i + .5) / 20 for i in range(20)]
    fig = go.Figure([go.Bar(x=ctr, y=cv["h0"], name="actual no", marker_color=NO), go.Bar(x=ctr, y=cv["h1"], name="actual yes", marker_color=YES)])
    fig.update_layout(barmode="stack", title=f"Score distribution – {BEST}", xaxis_title="Predicted probability of subscribing", yaxis_title="Clients", legend=dict(orientation="h", y=-.25))
    b.plotly_chart(fig.add_vline(x=metrics[BEST]["threshold"], line_dash="dash", annotation_text="cut-off").update_layout(height=400, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    st.subheader("Confusion matrices")
    cols = st.columns(3) + st.columns(3)
    for col, n in zip(cols, NAMES):
        fig = px.imshow(metrics[n]["cm"], x=["pred no", "pred yes"], y=["actual no", "actual yes"], text_auto=True, color_continuous_scale="Blues", title=n)
        fig.update_coloraxes(showscale=False); fig.update_layout(height=300, margin=dict(l=8, r=8, t=46, b=8)); col.plotly_chart(fig, use_container_width=True)
    st.subheader("What drives each model?")
    cols = st.columns(2) + st.columns(2)
    for col, n in zip(cols, ML):
        s = pd.DataFrame(curves[n]["imp"], columns=["feature", "value"]).iloc[::-1]
        fig = px.bar(s, x="value", y="feature", orientation="h", title=f"{n}: top {curves[n]['kind']}s", color=s["value"] > 0 if curves[n]["kind"] == "coefficient" else None,
                     color_discrete_map={True: YES, False: NO}, color_discrete_sequence=["#8FA3FF"]); fig.update_layout(showlegend=False, height=440, margin=dict(l=8, r=8, t=46, b=8)); col.plotly_chart(fig, use_container_width=True)
    nn = curves.get("Neural Network (MLP)")
    if nn and "train_loss" in nn:
        st.subheader("🧠 Neural network: architecture and learning curve")
        a, b = st.columns([1, 1.4])
        a.markdown(f"**Architecture** (one of {5} averaged networks): `{' → '.join(map(str, nn['arch']))}`  \n{nn['arch'][0]} inputs (encoded features) → hidden layers of {nn['arch'][1]} and {nn['arch'][2]} ReLU units → 1 sigmoid output (probability of subscribing).  \nTrained by back-propagation with the Adam optimiser and an L2 penalty; training stopped after **{nn['n_epochs']} epochs** when the validation score stopped improving (early stopping).")
        fig = go.Figure(); ep = list(range(1, len(nn["train_loss"]) + 1))
        fig.add_scatter(x=ep, y=nn["train_loss"], name="training loss (log-loss)", line=dict(color=C["Neural Network (MLP)"]))
        fig.add_scatter(x=ep, y=nn["val_acc"], name="validation accuracy", yaxis="y2", line=dict(color="#F5A524", dash="dot"))
        fig.update_layout(title="Learning curve", xaxis_title="Epoch", yaxis_title="Training loss", yaxis2=dict(title="Validation accuracy", overlaying="y", side="right", tickformat=".0%"), legend=dict(orientation="h", y=-.3))
        b.plotly_chart(fig.update_layout(height=330, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    m = metrics[BEST]; (tn, fp), (fn, tp) = m["cm"]; base = (tp + fn) / N; fl = tp + fp
    grade = "excellent" if m["roc"] >= .9 else "good" if m["roc"] >= .8 else "moderate" if m["roc"] >= .7 else "weak"
    st.subheader("How good are the predictions?")
    st.success(f"**Verdict: {grade}** (best learner: {BEST}, ROC-AUC {m['roc']:.3f})")
    st.markdown(f"""- Finds **{tp}** of **{tp+fn}** real subscribers ({m['rec']:.0%}) and misses {fn}.
- Flags {fl} clients ({fl/N:.1%} of the list); {m['prec']:.0%} of them subscribe, which is **{m['prec']/base:.1f}×** the {base:.1%} average.
- {fp} false alarms. Use the model to rank whom to call first; it is not reliable enough to decide about a single client.""")


# =============================================================== Predict
EX = {"yes": dict(age=34, job="student", marital="single", education="tertiary", default="no", balance=3200, housing="no", loan="no", contact="cellular", day=15, month="mar", campaign=2, pdays=90, previous=2, poutcome="success"),
      "no": dict(age=58, job="management", marital="married", education="tertiary", default="no", balance=2143, housing="yes", loan="no", contact="unknown", day=5, month="may", campaign=1, pdays=-1, previous=0, poutcome="unknown")}
INT = ["age", "balance", "day", "campaign", "pdays", "previous"]


def set_example(k):
    for a, v in EX[k].items(): st.session_state["w_" + a] = v


def gauge(title, p, th, color):
    f = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%", valueformat=".1f"), title=dict(text=title, font=dict(size=14)),
                               gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color), threshold=dict(line=dict(color="#ff4b4b", width=4), thickness=.85, value=th * 100))))
    f.update_layout(height=220, margin=dict(l=25, r=25, t=60, b=5)); return f


def predict():
    st.title("🔮 Predict a Client"); st.caption("Change any value — the result updates instantly. The red line on each gauge is the model's decision cut-off.")
    for k in RAW_FEATURES: st.session_state.setdefault("w_" + k, int(D[k]) if k in INT else D[k])
    for k in ("pdays",): st.session_state.setdefault("w_pdays", -1)
    b = st.columns([1.2, 1.2, 3]); b[0].button("✅ Load a likely-yes example", on_click=set_example, args=("yes",)); b[1].button("❌ Load a likely-no example", on_click=set_example, args=("no",))
    L, R = st.columns([1, 1.45], gap="large")
    with L:
        sel = lambda box, k, lab: box.selectbox(lab, cats[k], key="w_" + k)
        with st.expander("👤 Client", expanded=True):
            c1, c2 = st.columns(2); c1.number_input("Age", 18, 100, key="w_age"); sel(c2, "job", "Job"); sel(c1, "marital", "Marital status"); sel(c2, "education", "Education")
        with st.expander("💰 Finances", expanded=True):
            c1, c2 = st.columns(2); c1.number_input("Average yearly balance (€)", -20000, 200000, key="w_balance"); sel(c2, "default", "Credit in default?"); sel(c1, "housing", "Housing loan?"); sel(c2, "loan", "Personal loan?")
        with st.expander("📞 Campaign contact", expanded=True):
            c1, c2 = st.columns(2); sel(c1, "contact", "Contact type"); sel(c2, "poutcome", "Outcome of previous campaign"); c1.number_input("Last contact day of month", 1, 31, key="w_day"); sel(c2, "month", "Last contact month")
            c1.number_input("Contacts in this campaign", 1, 100, key="w_campaign"); c2.number_input("Contacts before this campaign", 0, 300, key="w_previous")
            c1.number_input("Days since last campaign contact (-1 = never)", -1, 1000, key="w_pdays")
    row = {k: st.session_state["w_" + k] for k in RAW_FEATURES}
    eng = add_engineered_features(pd.DataFrame([row]))[ENG_FEATURES]; r = eng.iloc[0].to_dict(); F, fired = forward_chain(rules, r)
    probs = {n: float(models[n]["pipeline"].predict_proba(eng)[0, 1]) for n in ML}
    votes = [probs[n] >= models[n]["threshold"] for n in ML] + ["Recommend" in F]
    with R:
        yes = sum(votes) > len(votes) / 2
        (st.success if yes else st.error)(f"### {sum(votes)} of {len(votes)} approaches predict **y = {1 if yes else 0}** — " + ("client is likely to SUBSCRIBE" if yes else "client is unlikely to subscribe"))
        g = st.columns(3) + st.columns(3)
        for col, n in zip(g, ML): col.plotly_chart(gauge(n, probs[n], models[n]["threshold"], C[n]), use_container_width=True)
        rec = votes[-1]
        g[len(ML)].markdown(f"<div class='kpi' style='margin-top:48px;text-align:center'><div class='l'>Logical rules</div><div class='v' style='color:{YES if rec else NO}'>Recommend(c) = {rec}</div><div class='s'>{len(F)} facts derived, {len(fired)} rules fired</div></div>", unsafe_allow_html=True)
        pipe = models["Logistic Regression"]["pipeline"]; prep = pipe.named_steps["prep"]; x = prep.transform(eng)[0]; w = pipe.named_steps["model"].coef_[0]; agg = {}
        for n, v in zip(prep.get_feature_names_out(), w * x): k = next((c for c in CAT if n.startswith(c + "_")), n); agg[k] = agg.get(k, 0) + v
        s = pd.Series(agg).sort_values(key=abs, ascending=False).head(8)[::-1]
        fig = px.bar(x=s.values, y=s.index, orientation="h", color=s.values > 0, color_discrete_map={True: YES, False: NO}, title="Why? Biggest pushes on the Logistic Regression score (green = towards yes)")
        fig.update_layout(showlegend=False, height=320, xaxis_title="Contribution to log-odds", yaxis_title=None, margin=dict(l=8, r=8, t=46, b=8)); st.plotly_chart(fig, use_container_width=True)
    a, b2 = st.columns(2)
    with a.expander("Forward chaining — facts derived and rules fired", expanded=True):
        st.write("**Facts:**", ", ".join(f"`{f}(c)`" for f in sorted(F)) or "none"); st.write("**Rules fired, in order:**", " → ".join(fired) or "none")
    with b2.expander("Backward chaining — prove a goal"):
        heads = sorted({x["head"] for x in rules}); goal = st.selectbox("Goal", heads, index=heads.index("Recommend")); st.code("\n".join(proof_lines(prove(rules, goal, r))), language=None)


# =============================================================== Batch
def score_frame(df):
    eng = add_engineered_features(df[RAW_FEATURES])[ENG_FEATURES]; out = df.copy()
    for n in ML:
        p = models[n]["pipeline"].predict_proba(eng)[:, 1]; k = n.split()[0]; out[f"P_{k}"] = p.round(4); out[f"pred_{k}"] = np.where(p >= models[n]["threshold"], "yes", "no")
    out["pred_Rules"] = ["yes" if "Recommend" in forward_chain(rules, x)[0] else "no" for x in eng.to_dict("records")]; return out


def batch():
    st.title("📂 Batch Prediction"); st.caption("Score many clients at once and see who to call first.")
    st.code(", ".join(RAW_FEATURES), language=None); st.download_button("Download empty template", pd.DataFrame(columns=RAW_FEATURES).to_csv(index=False), "template.csv")
    up = st.file_uploader("Upload a CSV (bank-full.csv format; `y` and `duration` optional)", type=["csv"]); sample = st.checkbox("…or try 500 sample clients from the dataset")
    df = pd.read_csv(up, sep=None, engine="python") if up else (data().drop(columns=["subscribed"]).sample(500, random_state=1) if sample else None)
    if df is None: return
    miss = [c for c in RAW_FEATURES if c not in df.columns]
    if miss: st.error(f"Missing columns: {miss}"); return
    out = score_frame(df); c = st.columns(1 + len(KS)); kpi(c[0], "Clients scored", f"{len(out):,}")
    for col, k in zip(c[1:], KS): kpi(col, f"{k} says yes", f"{(out['pred_' + k] == 'yes').sum():,}", f"{(out['pred_' + k] == 'yes').mean():.1%} of clients")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(out, x="P_Logistic", nbins=25, title="Distribution of subscribe probability (Logistic Regression)", color_discrete_sequence=[C["Logistic Regression"]]).update_layout(height=340, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    cnt = pd.DataFrame({"approach": KS, "yes": [(out["pred_" + k] == "yes").sum() for k in KS]})
    b.plotly_chart(px.bar(cnt, x="approach", y="yes", text="yes", title="Clients predicted to subscribe", color="approach", color_discrete_sequence=list(C.values())).update_layout(showlegend=False, height=340, margin=dict(l=8, r=8, t=46, b=8)), use_container_width=True)
    st.subheader("Top 15 clients to call first (ranked by the neural network)"); st.dataframe(out.sort_values("P_Neural", ascending=False).head(15)[["age", "job", "balance", "month", "poutcome"] + [f"P_{k}" for k in KS[:-1]] + ["pred_Rules"]], use_container_width=True)
    if "y" in df.columns:
        yt = df["y"].map({"no": 0, "yes": 1}).values; st.subheader("Check against your `y` column")
        st.dataframe(pd.DataFrame({k: dict(accuracy=accuracy_score(yt, out["pred_" + k] == "yes"), precision=precision_score(yt, out["pred_" + k] == "yes", zero_division=0), recall=recall_score(yt, out["pred_" + k] == "yes"), f1=f1_score(yt, out["pred_" + k] == "yes")) for k in KS}).T.round(3))
    st.download_button("Download all predictions", out.to_csv(index=False), "predictions.csv", type="primary")


# =============================================================== Logic
def logic():
    st.title("🧠 Logic & Rules"); c = kb["constants"]
    st.markdown("A **first-order-logic knowledge base** of 13 rules (c ranges over customers). **Forward chaining** starts from the client's facts and derives new facts until nothing changes; **backward chaining** starts from a goal such as `Recommend(c)` and proves it from the rules.")
    st.dataframe(pd.DataFrame([dict(Rule=r["id"], Stratum=r["s"], Logic=fol_rule(r)) for r in rules]), use_container_width=True, hide_index=True)
    st.caption(f"Constants learned from training data: months {c['months']}, jobs {c['jobs']}, balance ≥ {c['balance']:.2f}, campaign ≥ {c['campaign']:g}.")
    s = pd.DataFrame(kb["stats"]); s["rate_pct"] = s["rate"]
    fig = rate_bar(s.rename(columns={"atom": "fact"}), "fact", kb["base"], "Do the rules find real signal? Subscription rate when each derived fact is true (training data)")
    show(fig, 400)
    st.dataframe(s.assign(rate=s["rate"].map("{:.1%}".format), lift=(s["rate"] / kb["base"]).map("{:.2f}×".format))[["atom", "n", "rate", "lift"]].rename(columns=dict(atom="Derived fact", n="Clients", rate="Subscription rate", lift="Lift vs average")), use_container_width=True, hide_index=True)


# =============================================================== About
def about():
    st.title("ℹ️ About"); i = meta
    st.markdown(f"""**Goal.** Predict the target `y` — whether a bank client subscribes to a term deposit — from information available *before* the call.

**Data.** UCI Bank Marketing (`bank-full.csv`): {i['n_total']:,} clients after de-duplication, split 80:20 into {i['n_train']:,} training and {i['n_test']:,} test clients (stratified).

**Preprocessing.** Duplicates removed; `y` encoded yes=1/no=0; `pdays = -1` turned into a `previously_contacted` flag; binary columns to 0/1; other categoricals one-hot encoded ("unknown" kept); numeric features median-imputed, clipped at the training 1st/99th percentiles and standardised. Everything is fitted on the training set only.

**AI algorithms.** A **neural network** (ensemble of five multi-layer perceptrons, trained by back-propagation), plus Logistic Regression, Decision Tree and Random Forest. Each uses a decision threshold chosen from cross-validated training predictions (class weights balance the tree models and logistic regression; the networks rely on the tuned threshold).

**AI techniques.** Rule-based reasoning with forward chaining, and goal-driven backward chaining over a first-order-logic knowledge base.

**Limitations.** Call duration is excluded, which caps accuracy; the data come from one campaign; the model ranks clients well but misses roughly half of real subscribers, so it should guide call priorities, not individual decisions.""")
