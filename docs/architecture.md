# Architecture

```
data/raw/bank-full.csv ──► src/train.py ──► artifacts/ ──► src/dashboard/pages.py ──► app.py ──► browser
                              │                                   │
                              └──── src/preprocessing.py ◄────────┤
                                    src/reasoning.py     ◄────────┘
```

| Layer | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | Paths to data and artifacts |
| Preprocessing | `src/preprocessing.py` | Feature lists, `pdays` fix, clipper, `ColumnTransformer` |
| Neural network | `src/neural.py` | `MLPEnsemble`: 5 feed-forward networks (32→16 ReLU, Adam, L2, early stopping), probabilities averaged |
| Reasoning | `src/reasoning.py` | 13 first-order-logic rules, forward chaining, backward chaining (proof tree) |
| Training | `src/train.py` | Stratified split, four ML models (incl. the neural network), rule evaluation, writes `artifacts/` |
| Dashboard | `src/dashboard/pages.py` | Seven pages (Plotly charts, forms, batch scoring) |
| Entry point | `app.py` | Page config and left-side navigation |

## Artifacts (created by `python -m src.train`)
`models.joblib` (pipelines + decision cut-offs), `metrics.json`, `curves.json` (ROC/PR/gains/importance data),
`rules.json` (rules, constants, training statistics), `meta.json` (dropdown options, defaults, split sizes).

## Leakage controls
The 80:20 stratified split happens before anything is fitted. Encoders, clipping limits, scaler, rule constants and
decision thresholds are learned from the training set only; the test set is used once for the reported metrics.
`duration` is not an input because it is only known after the call.
