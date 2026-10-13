"""Trains 2 ML algorithms + evaluates a rule-based (forward chaining) system, then builds dashboard.html.
Run after the preprocessing notebook:  DATA_DIR=processed_data python dashboard_build.py"""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_curve, roc_curve)

class QuantileClipper(BaseEstimator, TransformerMixin):   # needed to unpickle preprocessor.joblib
    def __init__(s, lower=.01, upper=.99): s.lower = lower; s.upper = upper

D = os.environ.get("DATA_DIR", "processed_data"); OUT = os.environ.get("OUT", "dashboard.html")
P = lambda f: os.path.join(D, f)
cfg = json.load(open(P("preprocessing_config.json"))); prep = joblib.load(P("preprocessor.joblib"))
Xtr, Xte = pd.read_csv(P("X_train_processed.csv")), pd.read_csv(P("X_test_processed.csv"))
ytr, yte = pd.read_csv(P("y_train.csv")).iloc[:, 0].values, pd.read_csv(P("y_test.csv")).iloc[:, 0].values
tr, te = pd.read_csv(P("train_raw.csv")), pd.read_csv(P("test_raw.csv"))
mt = lambda y, h: dict(acc=accuracy_score(y, h), prec=precision_score(y, h, zero_division=0),
                       rec=recall_score(y, h), f1=f1_score(y, h), cm=confusion_matrix(y, h).tolist())

# ---------- AI algorithms 1 & 2 (trained/tuned on TRAIN only, evaluated once on TEST) ----------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
algos = {"Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=2000), {"C": [.01, .1, 1, 10]}),
         "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                           {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [20, 50, 100]})}
res, fit = {}, {}
for n, (m, g) in algos.items():
    gs = GridSearchCV(m, g, scoring="average_precision", cv=cv).fit(Xtr, ytr); b = gs.best_estimator_
    oof = cross_val_predict(b, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(ytr, oof)
    th = float(t[np.argmax(2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None))])  # threshold from train OOF
    pr = b.predict_proba(Xte)[:, 1]; f, tp, _ = roc_curve(yte, pr); ix = np.linspace(0, len(f) - 1, 60).astype(int)
    res[n] = dict(**mt(yte, (pr >= th).astype(int)), roc=roc_auc_score(yte, pr), pr=average_precision_score(yte, pr),
                  th=th, cv_pr=gs.best_score_, params=gs.best_params_,
                  curve=[[round(f[i], 4), round(tp[i], 4)] for i in ix])
    fit[n] = b

# ---------- AI techniques 1 & 2: rule-based reasoning on a first-order-logic KB (forward + backward chaining) ----------
base = tr.y_encoded.mean(); rt = lambda c: tr.groupby(c).y_encoded.mean()
HM = sorted(m for m, r in rt("month").items() if r > 2 * base)      # data-driven constants (TRAIN only)
HJ = sorted(j for j, r in rt("job").items() if r > 1.3 * base)
BH, CH = float(tr.balance.quantile(.75)), float(tr.campaign.quantile(.9))
R = [("R1", "PriorSuccess", [["poutcome", "==", "success"]], 0), ("R2", "HighSeason", [["month", "in", HM]], 0),
     ("R3", "TargetJob", [["job", "in", HJ]], 0), ("R4", "HighBalance", [["balance", ">=", BH]], 0),
     ("R5", "HasDebt", [["housing", "==", "yes"]], 0), ("R6", "HasDebt", [["loan", "==", "yes"]], 0),
     ("R7", "OverContacted", [["campaign", ">=", CH]], 0),
     ("R8", "Likely", ["PriorSuccess"], 1), ("R9", "Likely", ["HighSeason", "~HasDebt"], 1),
     ("R10", "Likely", ["TargetJob", "HighBalance"], 1), ("R11", "Unlikely", ["OverContacted"], 1),
     ("R12", "Unlikely", ["HasDebt", "~PriorSuccess", "~HighSeason"], 1), ("R13", "Recommend", ["Likely", "~Unlikely"], 2)]
rules = [dict(id=i, head=h, cond=c, s=s) for i, h, c, s in R]

def test(c, row, F):
    if isinstance(c, list):
        a, op, v = c; x = row[a]; return x == v if op == "==" else (x in v if op == "in" else x >= v)
    return (c[1:] not in F) if c[0] == "~" else c in F

def fc(row):  # forward chaining, stratified so negation-as-failure is sound
    F = set()
    for s in range(3):
        ch = True
        while ch:
            ch = False
            for r in rules:
                if r["s"] == s and r["head"] not in F and all(test(c, row, F) for c in r["cond"]): F.add(r["head"]); ch = True
    return F

Ftr, Fte = [fc(r) for r in tr.to_dict("records")], [fc(r) for r in te.to_dict("records")]
res["Rule-based (forward chaining)"] = dict(**mt(yte, np.array(["Recommend" in F for F in Fte], int)), roc=None, pr=None, th=None, cv_pr=None, params={})
stats = []
for a in ["PriorSuccess", "HighSeason", "TargetJob", "HighBalance", "HasDebt", "OverContacted", "Likely", "Unlikely", "Recommend"]:
    m = np.array([a in F for F in Ftr]); stats.append(dict(atom=a, n=int(m.sum()), rate=float(ytr[m].mean()) if m.any() else 0))

# ---------- export for the in-browser dashboard ----------
nm = prep.named_transformers_["num"]; lo, hi = nm.named_steps["clip"].lower_, nm.named_steps["clip"].upper_
mu, sd = nm.named_steps["scale"].mean_, nm.named_steps["scale"].scale_
nums = {c: [lo[i], hi[i], mu[i], sd[i]] for i, c in enumerate(cfg["num_cols"])}; feat = []
for n in cfg["feature_names_processed"]:
    if n in nums: feat.append(dict(t="num", c=n, p=nums[n]))
    elif n in cfg["binary_cols"]: feat.append(dict(t="bin", c=n))
    elif n in cfg["flag_cols"]: feat.append(dict(t="flag"))
    else:
        c = next(c for c in cfg["cat_cols"] if n.startswith(c + "_")); feat.append(dict(t="cat", c=c, v=n[len(c) + 1:]))

def vec(row):  # Python twin of the JS feature builder -> verifies the export reproduces the real pipeline
    row = dict(row, previously_contacted=int(row["pdays"] > 0 or row["previous"] > 0))
    return [(min(max(row[f["c"]], f["p"][0]), f["p"][1]) - f["p"][2]) / f["p"][3] if f["t"] == "num" else
            float(row[f["c"]] == "yes") if f["t"] == "bin" else float(row["previously_contacted"]) if f["t"] == "flag"
            else float(row[f["c"]] == f["v"]) for f in feat]
chk = te[te.pdays.between(1, 10_000) | (te.pdays == 0)].head(200)
err = max(np.abs(np.array(vec(r)) - Xte.loc[i].values).max() for i, r in chk.to_dict("index").items())
print("feature-export max abs error vs sklearn pipeline:", err); assert err < 1e-6

lr, dt = fit["Logistic Regression"], fit["Decision Tree"].tree_
top = sorted(zip(Xtr.columns, lr.coef_[0]), key=lambda z: -abs(z[1]))[:10]
ui_cat = {c: sorted(tr[c].unique().tolist()) for c in ["job", "marital", "education", "contact", "month", "poutcome"]}
ui_cat["month"] = [m for m in "jan feb mar apr may jun jul aug sep oct nov dec".split() if m in ui_cat["month"]]
try: dups = int(pd.read_csv(os.environ["RAW"], sep=";").duplicated().sum())
except Exception: dups = None
data = dict(models=res, rules=rules, stats=stats, base=float(base), feat=feat, top=top,
            lr=dict(w=lr.coef_[0].tolist(), b=float(lr.intercept_[0])),
            dt=dict(f=dt.feature.tolist(), th=dt.threshold.tolist(), l=dt.children_left.tolist(), r=dt.children_right.tolist(),
                    p=(dt.value[:, 0, 1] / dt.value[:, 0, :].sum(1)).tolist()),
            ui=dict(cat=ui_cat, defaults={**{c: tr[c].mode()[0] for c in ui_cat}, **{c: float(tr[c].median()) for c in ["age", "balance", "day", "campaign"]}, "pdays": 0, "previous": 0,
                    "default": "no", "housing": "yes", "loan": "no"}),
            info=dict(n_train=len(tr), n_test=len(te), n_raw_feat=len(Xtr.columns) and len(cfg["feature_cols_raw"]), n_proc=Xtr.shape[1],
                      pos_train=float(ytr.mean()), pos_test=float(yte.mean()), dups=dups, hm=HM, hj=HJ, bh=BH, ch=CH))
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")).read()
open(OUT, "w").write(html.replace("__DATA__", json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
print({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items() if m in ("acc", "prec", "rec", "f1", "roc", "pr")} for k, d in res.items()})
print("saved", OUT, os.path.getsize(OUT) // 1024, "KB")
