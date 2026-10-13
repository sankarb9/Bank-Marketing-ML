"""Trains the models and writes everything the web app needs into ./artifacts.
Run from the project root:  python -m src.train   (the app also runs it automatically if artifacts are missing)."""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_curve)
from src.config import ARTIFACTS, DATA_PATH
from src.preprocessing import CAT, ENG_FEATURES, add_engineered_features, make_preprocessor
from src.neural import MLPEnsemble
from src.reasoning import forward_chain, make_rules

DATA = os.environ.get("DATA_PATH") or DATA_PATH
ARTIFACTS.mkdir(exist_ok=True)
df = pd.read_csv(DATA, sep=";").drop_duplicates().reset_index(drop=True)          # 1. remove duplicates
df["y_encoded"] = df["y"].map({"no": 0, "yes": 1})                                  # 2. encode target
df = add_engineered_features(df)
train, test = train_test_split(df, test_size=0.2, stratify=df["y_encoded"], random_state=42)  # 3. stratified 80:20
Xtr, Xte, ytr, yte = train[ENG_FEATURES], test[ENG_FEATURES], train["y_encoded"].values, test["y_encoded"].values

prep = make_preprocessor().fit(Xtr)                                                 # 4. fit preprocessing on TRAIN only
Xtr_p, Xte_p = prep.transform(Xtr), prep.transform(Xte)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
algos = {"Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
         "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=42),
         "Neural Network (MLP)": MLPEnsemble(n_members=5, hidden=(32, 16), alpha=0.1, random_state=42),
         "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42)}
mt = lambda y, h: dict(acc=accuracy_score(y, h), prec=precision_score(y, h, zero_division=0), rec=recall_score(y, h),
                       f1=f1_score(y, h), cm=confusion_matrix(y, h).tolist())
models, metrics = {}, {}
for name, m in algos.items():
    oof = cross_val_predict(m, Xtr_p, ytr, cv=cv, method="predict_proba")[:, 1]       # threshold from TRAIN OOF predictions
    p, r, t = precision_recall_curve(ytr, oof)
    th = float(t[np.argmax(2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None))])
    m.fit(Xtr_p, ytr)
    pr = m.predict_proba(Xte_p)[:, 1]
    metrics[name] = dict(**mt(yte, (pr >= th).astype(int)), roc=roc_auc_score(yte, pr),
                         pr=average_precision_score(yte, pr), threshold=th)
    models[name] = dict(pipeline=Pipeline([("prep", prep), ("model", m)]), threshold=th)

base = ytr.mean(); rate = lambda c: train.groupby(c)["y_encoded"].mean()          # rule constants from TRAIN only
HM = sorted(k for k, v in rate("month").items() if v > 2 * base)
HJ = sorted(k for k, v in rate("job").items() if v > 1.3 * base)
BH, CH = float(train["balance"].quantile(.75)), float(train["campaign"].quantile(.9))
rules = make_rules(HM, HJ, BH, CH)
rec = np.array(["Recommend" in forward_chain(rules, r)[0] for r in Xte.to_dict("records")], int)
metrics["Rule-based (forward chaining)"] = dict(**mt(yte, rec), roc=None, pr=None, threshold=None)


# ---- curve data for the dashboard charts (all from TEST predictions / TRAIN statistics) ----
sub = lambda n, k: np.unique(np.linspace(0, n - 1, k).astype(int))
fnames = prep.get_feature_names_out(); curves = {}
for name, m in algos.items():
    pr = m.predict_proba(Xte_p)[:, 1]
    f, t, _ = roc_curve(yte, pr); ix = sub(len(f), 80)
    pp, rr, _ = precision_recall_curve(yte, pr); iy = sub(len(pp), 80)
    o = np.argsort(-pr); cum = np.cumsum(yte[o]) / yte.sum(); gx = sub(len(o), 60)
    if hasattr(m, "coef_"): imp, kind = m.coef_[0], "coefficient"
    elif hasattr(m, "feature_importances_"): imp, kind = m.feature_importances_, "importance"
    else:                                                                             # neural network: permutation importance (PR-AUC drop) on a train sample
        ix_s = np.random.RandomState(0).choice(len(ytr), 6000, replace=False)
        imp, kind = permutation_importance(m, Xtr_p[ix_s], ytr[ix_s], scoring="average_precision", n_repeats=3, random_state=0).importances_mean, "permutation importance"
    top = np.argsort(-np.abs(imp))[:15]
    curves[name] = dict(roc=[f[ix].round(4).tolist(), t[ix].round(4).tolist()], pr=[rr[iy].round(4).tolist(), pp[iy].round(4).tolist()],
                        gain=[((gx + 1) / len(o)).round(4).tolist(), cum[gx].round(4).tolist()],
                        h1=np.histogram(pr[yte == 1], bins=20, range=(0, 1))[0].tolist(), h0=np.histogram(pr[yte == 0], bins=20, range=(0, 1))[0].tolist(),
                        imp=[[str(fnames[i]), float(imp[i])] for i in top], kind=kind)
    if hasattr(m, "models_"):                                                         # learning curve of the first network (loss + validation accuracy per epoch)
        n0 = m.models_[0]; curves[name]["train_loss"] = [round(float(v), 5) for v in n0.loss_curve_]; curves[name]["val_acc"] = [round(float(v), 5) for v in n0.validation_scores_]
        curves[name]["arch"] = m.architecture(Xtr_p.shape[1]); curves[name]["n_epochs"] = int(n0.n_iter_)
json.dump(curves, open(ARTIFACTS / "curves.json", "w"))
Ftr = [forward_chain(rules, r)[0] for r in Xtr.to_dict("records")]; stats = []
for a in ["PriorSuccess", "HighSeason", "TargetJob", "HighBalance", "HasDebt", "OverContacted", "Likely", "Unlikely", "Recommend"]:
    mk = np.array([a in F for F in Ftr]); stats.append(dict(atom=a, n=int(mk.sum()), rate=float(ytr[mk].mean()) if mk.any() else 0.0))
order = "jan feb mar apr may jun jul aug sep oct nov dec".split()
cats = {c: sorted(train[c].unique().tolist()) for c in CAT + ["default", "housing", "loan"]}
cats["month"] = [m for m in order if m in cats["month"]]
defaults = {**{c: train[c].mode()[0] for c in cats}, **{c: float(train[c].median()) for c in ["age", "balance", "day", "campaign"]},
            "pdays": -1, "previous": 0}
joblib.dump(models, ARTIFACTS / "models.joblib")
json.dump(dict(rules=rules, stats=stats, base=float(base), constants=dict(months=HM, jobs=HJ, balance=BH, campaign=CH)), open(ARTIFACTS / "rules.json", "w"), indent=1)
json.dump(metrics, open(ARTIFACTS / "metrics.json", "w"), indent=1, default=float)
json.dump(dict(cats=cats, defaults=defaults, n_train=len(train), n_test=len(test), pos_rate_test=float(yte.mean()),
               n_total=len(df)), open(ARTIFACTS / "meta.json", "w"), indent=1, default=float)
for k, v in metrics.items():
    print(f"{k:32s} acc={v['acc']:.3f} prec={v['prec']:.3f} rec={v['rec']:.3f} f1={v['f1']:.3f} roc={v['roc'] if v['roc'] is None else round(v['roc'],3)}")
print("artifacts written to", ARTIFACTS)
