# Bank Marketing – Term-Deposit Prediction

Predicts the target **y** (will a client subscribe to a term deposit?) with four machine-learning algorithms
(a neural network [MLP ensemble], Logistic Regression, Decision Tree, Random Forest) and rule-based reasoning (first-order logic, forward and backward chaining),
presented in an interactive Streamlit dashboard with a left-side menu.

## Structure
```
bank-marketing-project/
├── app.py                     Streamlit entry point (navigation menu)
├── requirements.txt           Python packages
├── Dockerfile / .dockerignore Container deployment
├── .streamlit/config.toml     Theme and server settings
├── data/raw/bank-full.csv     Dataset (UCI Bank Marketing)
├── src/
│   ├── config.py              Paths
│   ├── preprocessing.py       Features, encoding, scaling pipeline
│   ├── neural.py              Neural network: ensemble of multi-layer perceptrons
│   ├── reasoning.py           FOL rules, forward/backward chaining
│   ├── train.py               Trains models -> artifacts/
│   └── dashboard/pages.py     The seven dashboard pages
├── artifacts/                 Saved models and chart data (generated)
├── notebooks/
│   ├── 01_preprocessing.ipynb Cleaning, encoding, split, plots, report
│   └── 02_model_training.ipynb Model comparison, tuning, evaluation (incl. gradient boosting reference)
├── static_dashboard/          Single-file HTML dashboard (works on Vercel/GitHub Pages)
├── tests/                     pytest smoke tests
└── docs/architecture.md       Design and data flow
```

## Run locally
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                  # http://localhost:8501
```
The first start trains the models automatically (about 1.5 minutes). To train manually: `python -m src.train`.

## Pages
Overview · Data Explorer · Model Performance · Predict a Client · Batch Prediction · Logic & Rules · About

## Tests
```bash
pip install pytest
pytest -q
```

## Deploy
- **Docker:** `docker build -t bank-dashboard . && docker run -p 8501:8501 bank-dashboard`
- **Streamlit Community Cloud:** push the repo to GitHub, main file `app.py` (keep `data/raw/bank-full.csv`).
- **Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io:** use the Dockerfile.
- **Vercel / static hosting:** only the `static_dashboard/` page works there; Streamlit needs a running Python server.

## Notebooks
Run `notebooks/01_preprocessing.ipynb` first (creates `processed_data/`), then `02_model_training.ipynb`, from the `notebooks/` folder or in Google Colab.

## Static dashboard
`static_dashboard/dashboard.html` opens in any browser. Rebuild it after preprocessing with
`DATA_DIR=notebooks/processed_data python static_dashboard/build_static_dashboard.py`.

## Notes
`duration` is excluded because it is only known after the call (data leakage). `pdays = -1` means never contacted.
Saved models depend on the scikit-learn version that trained them: retrain after upgrading.
#

# Bank Marketing Subscription Prediction

## Project Overview

This project predicts whether a bank client will subscribe to a term deposit (`y = yes/no`) using information available **before the marketing call**. It combines machine learning models with rule-based reasoning and first-order logic to support data-driven marketing decisions.

## Dataset

- **Source:** UCI Bank Marketing Dataset (`bank-full.csv`)
- **Total clients:** 45,211 after de-duplication
- **Training set:** 36,168 clients (80%)
- **Test set:** 9,043 clients (20%)
- **Split strategy:** Stratified train-test split
- **Target variable:** `y` — subscription to a term deposit (`yes = 1`, `no = 0`)

## Data Preprocessing

The following preprocessing steps are applied:

- Removed duplicate records.
- Encoded the target variable as binary values.
- Converted `pdays = -1` into a `previously_contacted` indicator.
- Converted binary categorical columns into `0/1` values.
- Applied one-hot encoding to other categorical features, retaining the `unknown` category.
- Imputed missing numeric values using the median.
- Clipped numeric features at the training-set 1st and 99th percentiles.
- Standardised numeric features.
- Fitted preprocessing transformations using the training set only to prevent data leakage.

## AI Algorithms

The project implements four machine learning approaches:

1. **Neural Network:** An ensemble of five multilayer perceptrons (MLPs), trained using backpropagation.
2. **Logistic Regression:** A classification model that estimates the likelihood of subscription.
3. **Decision Tree:** A tree-based model that learns interpretable decision rules.
4. **Random Forest:** An ensemble of decision trees that captures more complex patterns.

Decision thresholds are selected using cross-validated training predictions. Class weights are used to address class imbalance in Logistic Regression and the tree-based models, while the neural network ensemble relies on threshold tuning.

## AI Techniques

### 1. Rule-Based Reasoning

The system uses predefined rules to derive conclusions from client information. Forward chaining applies available facts and rules to infer additional facts until a conclusion is reached or no further rules can be applied.

### 2. First-Order Logic

A first-order-logic knowledge base represents facts and relationships about clients and subscription outcomes.

### 3. Backward Chaining

Backward chaining uses a goal-driven approach. It begins with a query or goal and works backward through relevant rules to determine whether the available facts support the conclusion.

These techniques complement machine learning by providing a structured, logic-based approach to reasoning.

## Limitations

- **Call duration excluded:** Call duration is not used because it is unavailable before the call. This limits the maximum predictive performance but makes the model more useful for pre-call targeting.
- **Single-campaign dataset:** The data come from one marketing campaign, so results may not generalise to other campaigns, banks, or customer populations.
- **Missed subscribers:** Although the model ranks clients effectively, it misses roughly half of actual subscribers at the selected operating threshold.
- **Decision support only:** Predictions should help prioritise marketing calls rather than determine individual customer decisions automatically.

## Conclusion

This project combines neural networks, traditional machine learning algorithms, rule-based reasoning, forward chaining, backward chaining, and first-order logic to predict term-deposit subscriptions.

The primary objective is to support more informed marketing-call prioritisation using information available before contact, while recognising the limitations of the dataset and predictive models.
