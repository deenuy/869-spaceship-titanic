# 869-spaceship-titanic
Predict whether passengers aboard the Spaceship Titanic were transported to an alternate dimension using structured data and advanced machine learning techniques. The competition is hosted on Kaggle and mimics an interstellar disaster scenario where nearly 13,000 passengers' fate must be determined by data.

## 🚀 Features
- 🔍 Exploratory Data Analysis (EDA) and imbalanced data handling
- 🧱 Feature engineering including derived features, group aggregations, and embeddings
- 🔄 Model experimentation pipeline using Optuna, GridSearchCV, and ensemble methods
- 🪄 Non-tree models explored: SVM, Logistic Regression, KNN, Neural Networks
- 📊 Cross-validation vs Leaderboard tracking with Weights & Biases (W&B)
- 💻 GPU-accelerated training via Google Colab Pro+
- 🔁 Versioned experimentation with GitHub integration
- 🧪 Error analysis and explainability using SHAP, confusion matrices, and misclassified examples

## 🎯 Goal
Maximize leaderboard accuracy on the Kaggle competition by systematically exploring:
- Multiple ML algorithms (XGBoost, LGBM, CatBoost, etc.)
- Robust hyperparameter tuning
- Data-centric AI practices

## 📁 Project Structure

```bash
  869-spaceship-titanic/
  ├── data/                                                 # Raw, processed, and submission datasets
  │   ├── raw/                                              # Unmodified input datasets from Kaggle
  │   ├── processed/                                        # Cleaned, engineered, train/test split
  │   └── submission/                                       # Final CSV submissions for leaderboard
  ├── notebooks/                                            # Chronological modeling pipeline notebooks
  │   ├── 01-eda-initial-analysis.ipynb                     # Visual and statistical EDA to uncover survival patterns
  │   ├── 02-feature-engineering-v3.ipynb                   # Full-scale feature engineering with preprocessing
  │   ├── 03-baseline-model-v2.ipynb                        # Baseline models (LogReg, Trees) without tuning
  │   ├── 04-feature-imp-analysis-and-selection.ipynb       # Feature importance + selection using RF, MI, Corr
  │   ├── 05-baseline-model-best-feats.ipynb                # Baseline model using selected top features
  │   ├── 05-hyperparameter-opt-random-search-v1.ipynb      # RandomizedSearchCV tuning (wide exploration)
  │   ├── 06-hyperparameter-opt-grid-search-v1.ipynb        # GridSearchCV tuning (focused search space)
  │   ├── 06-hyperparameter-opt-TPE-optune-v1.ipynb         # Bayesian tuning with Optuna TPE + early pruning
  │   └── 07-final-evaluation-submission.ipynb              # Final model evaluation and Kaggle submission output
  ├── models/                                               # Serialized models and tuning artifacts
  ├── src/                                                  # Modularized source code for reusability
  ├── pyproject.toml                                        # Optional: Structured project metadata + build config
  ├── requirements.txt                                      # Fixed list of packages and versions for reproducibility
  ├── README.md                                             # Project overview, instructions, and team details
  └── .gitignore                                            # Prevents committing data, checkpoints, and cache

```


---

## ⚙️ Quickstart: Setup with [`uv`](https://github.com/astral-sh/uv)

> `uv` is a fast Python package manager and environment tool — a modern replacement for pip + virtualenv.

### ✅ Step 1: Install `uv` (one-time setup)

```bash
pip install uv
```

### ✅ Step 2: Clone the repo

```bash
git clone https://github.com/your-username/869-spaceship-titanic.git
cd 869-spaceship-titanic
```

### ✅ Step 3: Create isolated environment and install project

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### ✅ Optional: Lock Dependencies for Reproducibility
Machine learning experiments can break when package versions change unexpectedly. Locking dependencies ensures the exact same environment is used — down to the patch level — preserving reproducibility for future reruns, submissions, or sharing results with others.

To capture a simple snapshot:
```bash
uv pip freeze > requirements.txt
```

For full reproducibility (best for CI builds):
```bash
uv pip install uv-pip-tools
uv pip-compile pyproject.toml
```

> This creates a lockfile with fully pinned versions based on your `pyproject.toml`, ensuring consistent, conflict-free builds on any machine.
---

## 🧪 Running Experiments

- Start with the baseline notebooks in `/notebooks`
- Modular code lives in `/src` for:
    - Preprocessing
    - Feature engineering
    - Training
    - Tuning (Optuna / GridSearchCV)
    - Evaluation
- Track your metrics and experiments using [Weights & Biases](https://wandb.ai)

## 🌐 Colab GPU Setup (Optional)

If you're using [Google Colab Pro+](https://colab.research.google.com/):
- Upload this repo to GitHub
- Clone in Colab:
    ```bash
    !git clone https://github.com/deenuy/869-spaceship-titanic.git
    %cd 869-spaceship-titanic
    !pip install -e .
    ```

## 📚 References

- [Spaceship Titanic Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- [Optuna Documentation](https://optuna.org/)
- [SHAP Explainability](https://shap.readthedocs.io/)
- [W&B Experiment Tracking](https://wandb.ai)

## 📄 License
This project is for educational purposes under the Smith School of Business, Queen’s University.