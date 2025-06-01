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
    ├── data/               # Raw and processed datasets
    ├── notebooks/          # EDA, baseline models, tuning, and final reports
    ├── models/             # Saved model artifacts (e.g., joblib, ONNX, etc.)
    ├── src/                # Python package for modular pipeline code
    │   ├── __init__.py
    │   ├── data_prep.py
    │   ├── features.py
    │   ├── models.py
    │   ├── tuning.py
    │   └── explain.py
    ├── pyproject.toml      # Project metadata and dependencies
    ├── requirements.txt    # Locked dependencies (optional)
    ├── README.md           # You're here
    └── .gitignore
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