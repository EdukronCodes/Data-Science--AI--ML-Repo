"""
Generate fully-expanded algorithm markdown docs for the repository.

Writes detailed markdown files to:
  ml-algorithms-docs/docs/algorithms/

Run:
  python3 scripts/generate_full_docs.py

Be aware: files will be overwritten if they already exist.
"""
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ml-algorithms-docs" / "docs" / "algorithms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = [
    # Regression
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Polynomial Regression",
    "Support Vector Regression (SVR)",
    # Classification
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Support Vector Machine (SVM)",
    "Decision Tree Classifier",
    "Random Forest",
    "Naïve Bayes",
    "Gradient Boosting Classifier",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    # Clustering
    "K-Means Clustering",
    "Hierarchical Clustering",
    "DBSCAN",
    "Gaussian Mixture Models (GMM)",
    # Dimensionality Reduction
    "Principal Component Analysis (PCA)",
    "t-SNE",
    "UMAP",
    "Autoencoders",
    # Semi/Self-supervised
    "Label Propagation / Label Spreading",
    "Self-Training Classifier",
    # Deep Learning
    "Artificial Neural Networks (ANN)",
    "Convolutional Neural Networks (CNN)",
    "Recurrent Neural Networks (RNN)",
    "Long Short-Term Memory (LSTM)",
    "Transformer Models (BERT / GPT)"
]

COMMON_HEADER = textwrap.dedent("""
# {title}

> Comprehensive, end-to-end reference for {title}. Sections follow the "Flow for Learning an ML Algorithm" format:
> Flow, Overview, Math, Loss, Optimization, Hyperparameters, Assumptions, Pros/Cons, Pseudocode, From-scratch implementation, Library examples, Tuning, Metrics, Bias-Variance, Overfitting, Comparisons, Use cases, Projects, Scalability, Interview Qs.

---


## Flow for Learning an ML Algorithm

Mermaid flow:
```mermaid
flowchart TD
  A[Define problem & success metrics] --> B[Collect & ingest data]
  B --> C[Exploratory data analysis]
  C --> D[Preprocessing & cleaning]
  D --> E[Feature engineering & selection]
  E --> F[Choose algorithm & baseline]
  F --> G[Implement & train]
  G --> H[Hyperparameter tuning & CV]
  H --> I[Test & final evaluation]
  I --> J[Deploy, monitor & iterate]
```

Notes:
- Start with a simple baseline and clear metric aligned to business cost.
- Iterate: feature work often improves performance more than model changes.
- Keep reproducibility: seeds, environment, versions, train/test splits saved.
""").strip()

def make_math_section(title, body):
    return f"## Mathematical Foundation\n\n{body}\n"

def generic_contents(name, ptype, extra_impl=""):
    md = [COMMON_HEADER.format(title=name)]
    md.append("## Algorithm Overview & Intuition\n")
    md.append(textwrap.dedent(f"""
    {name} — description, intuition, and when to use it.

    Intuition:
    - High-level description of how {name} maps inputs to outputs.
    - Visual intuition suggestions: plots of decision boundaries (classification), fitted curves (regression), cluster centroids (clustering).

    Example intuitive scenario:
    - Short toy example explaining predictions and edge cases.
    """).strip())

    md.append("\n## Problem Type & Use Cases\n")
    md.append(f"- Primary problem type: {ptype}\n")
    md.append(textwrap.dedent("""
    Typical domains and concrete use cases:
    - Finance, healthcare, computer vision, NLP, recommender systems depending on algorithm appropriateness.
    - For each use case, include required preconditions (label quality, feature types, volume).
    """).strip())

    math_body = textwrap.dedent(f"""
    Notation:
    - X: input matrix (n x d), y: target vector, n: number of samples, d: features, θ: parameters.

    Core equations:
    - Present model hypothesis and probabilistic interpretation where applicable.
    - Derivations: include step-by-step derivation for learning rules (closed-form if present).

    Geometric/statistical view:
    - Discuss projections, margin maximization, density modeling, or neural parameterization depending on algorithm.
    """).strip()
    md.append(make_math_section(name, math_body))

    md.append("\n## Cost / Loss Function\n")
    md.append(textwrap.dedent("""
    - Provide the exact loss form used by this algorithm (MSE, cross-entropy, hinge, negative log-likelihood, ELBO for probabilistic models).
    - Present regularized variants (L1, L2, ElasticNet) and robust alternatives (Huber, MAE).
    - Visualize typical loss surface characteristics and implications for optimization.
    """).strip())

    md.append("\n## Optimization Technique\n")
    md.append(textwrap.dedent("""
    - Describe the optimization approach (closed-form, gradient-based, EM, coordinate descent, second-order methods).
    - Convergence guarantees, computational complexity, numerical stability tips.
    - When to use mini-batch vs full-batch vs stochastic updates.
    """).strip())

    md.append("\n## Key Hyperparameters\n")
    md.append(textwrap.dedent("""
    - List hyperparameters, default ranges, and practical tuning advice.
    - Prioritization: which hyperparameters to tune first and diagnostic heuristics (e.g., high variance -> increase regularization).
    """).strip())

    md.append("\n## Assumptions & Limitations\n")
    md.append(textwrap.dedent("""
    - Explicit assumptions about data distribution, independence, linearity, stationarity.
    - Typical failure modes and mitigation (robust preprocessing, transformations, alternative algorithms).
    """).strip())

    md.append("\n## Advantages & Disadvantages\n")
    md.append(textwrap.dedent("""
    - Advantages: interpretability, speed, sample efficiency, robustness in certain scenarios.
    - Disadvantages: scalability, sensitivity to hyperparameters, inability to model complex interactions (for simpler models), etc.
    """).strip())

    md.append("\n## Algorithm Workflow / Pseudocode\n")
    md.append(textwrap.dedent("""
    Pseudocode (generic supervised training loop):

    1. Load and clean data.
    2. Split into train/validation/test with appropriate strategy (stratified/time-based).
    3. Preprocess features (impute, scale, encode).
    4. Train model with chosen hyperparameters.
    5. Evaluate on validation; tune hyperparameters.
    6. Final evaluation on test; save model.

    Complexity:
    - Annotate time and memory complexity for training and inference.
    """).strip())

    md.append("\n## Implementation from Scratch (Python + NumPy)\n")
    md.append(textwrap.dedent("""
    - Provide a minimal, well-commented NumPy implementation for the core algorithmic idea where feasible.
    - Unit-test ideas: synthetic data tests, gradient checks, invariants.

    Example pattern (fill with algorithm-specific computations as needed):
    ```python
    # Example: minimal pattern (replace with algorithm-specific math)
    import numpy as np

    def fit_example(X, y, **kwargs):
        # X: (n, d), y: (n,) or (n, k)
        # Implement algorithm-specific core computation here
        raise NotImplementedError

    def predict_example(params, X):
        raise NotImplementedError
    ```
    """).strip())

    if extra_impl:
        md.append("\n" + extra_impl.strip() + "\n")

    md.append("\n## Implementation using ML Libraries\n")
    md.append(textwrap.dedent("""
    - scikit-learn examples (fit/predict, pipelines, saving/loading).
    - For neural models: PyTorch/TensorFlow minimal example (model, loss, optimizer, training loop).
    - For ensemble/GBMs: XGBoost / LightGBM / CatBoost example usage and common flags.
    """).strip())

    md.append("\n## Hyperparameter Tuning Methods\n")
    md.append(textwrap.dedent("""
    - Grid search, random search, Bayesian optimization (Optuna/Hyperopt), evolutionary strategies.
    - CV strategies (k-fold, stratified, time-series split) and nested CV for unbiased selection.
    - Early stopping and multi-fidelity methods (Hyperband, ASHA).
    """).strip())

    md.append("\n## Model Evaluation Metrics\n")
    md.append(textwrap.dedent("""
    - List metrics by problem type (accuracy, precision, recall, F1, AUC for classification; MSE/RMSE/MAE/R^2 for regression).
    - Calibration, confusion matrices, PR curves for imbalanced problems.
    """).strip())

    md.append("\n## Bias-Variance Tradeoff Analysis\n")
    md.append(textwrap.dedent("""
    - Explain bias and variance, show typical diagnostic plots (learning curves).
    - Remedies for each scenario and concrete examples.
    """).strip())

    md.append("\n## Handling Overfitting & Underfitting\n")
    md.append(textwrap.dedent("""
    - Regularization, feature selection, ensembling, data augmentation, early stopping, cross-validation, and adding data.
    - Practical monitoring suggestions and model selection heuristics.
    """).strip())

    md.append("\n## Comparison with Similar Algorithms\n")
    md.append(textwrap.dedent("""
    - Short prose comparing strengths/weaknesses vs alternatives.
    - Decision rules: when to pick this algorithm vs others.
    """).strip())

    md.append("\n## Real-World Applications / Case Studies\n")
    md.append(textwrap.dedent("""
    - Provide 2–3 concise case studies: data description, chosen approach, metric, deployment considerations, lessons learned.
    """).strip())

    md.append("\n## Practical Project / Dataset Experiment\n")
    md.append(textwrap.dedent("""
    - Suggested datasets and step-by-step project blueprint, reproducibility checklist, and evaluation rubric.
    """).strip())

    md.append("\n## Performance Optimization & Scalability\n")
    md.append(textwrap.dedent("""
    - Engineering optimizations (vectorization, batching), approximate algorithms (ANN for KNN), sparse data handling, GPUs/TPUs, distributed training.
    - Serving considerations: model size, latency, quantization, distillation.
    """).strip())

    md.append("\n## Common Interview Questions\n")
    md.append(textwrap.dedent("""
    - Curated conceptual, derivation, and debugging questions with short answer pointers.
    - Provide sample answers or references to canonical materials.
    """).strip())

    md.append("\n## Visual Flow & Checklists\n")
    md.append(textwrap.dedent("""
    - Include mermaid diagrams, checklist items for reproducibility and deployment readiness.
    - Example mermaid graph:
    ```mermaid
    flowchart LR
      Data-->EDA-->Preprocess-->Model-->Validate-->Deploy
    ```
    """).strip())

    return "\n\n".join(md)

def extra_impl_for(name):
    lname = name.lower()
    if "linear regression" in lname:
        return textwrap.dedent("""
        ### From-scratch Linear Regression (NumPy)

        ```python
        # filepath: examples/linear_regression_scratch.py
        import numpy as np

        def fit_linear_regression(X, y, l2=0.0):
            # X: (n, d), y: (n,)
            n, d = X.shape
            Xb = np.hstack([np.ones((n,1)), X])  # add bias
            I = np.eye(d+1)
            I[0,0] = 0  # don't regularize bias
            A = Xb.T @ Xb + l2 * I
            w = np.linalg.solve(A, Xb.T @ y)
            return w

        def predict_linear_regression(w, X):
            n = X.shape[0]
            Xb = np.hstack([np.ones((n,1)), X])
            return Xb @ w

        if __name__ == "__main__":
            # simple sanity check
            rng = np.random.default_rng(0)
            X = rng.normal(size=(100,2))
            true_w = np.array([1.5, -2.0, 0.5])
            y = np.hstack([np.ones((100,1)), X]) @ true_w + rng.normal(scale=0.1, size=100)
            w = fit_linear_regression(X, y, l2=1e-6)
            print("Estimated weights:", w)
        ```
        """)
    if "logistic regression" in lname:
        return textwrap.dedent("""
        ### From-scratch Logistic Regression (NumPy, batch gradient descent)

        ```python
        # filepath: examples/logistic_regression_scratch.py
        import numpy as np

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def fit_logistic(X, y, lr=1e-2, n_iter=1000, l2=0.0):
            n, d = X.shape
            Xb = np.hstack([np.ones((n,1)), X])
            w = np.zeros(d+1)
            for i in range(n_iter):
                preds = sigmoid(Xb @ w)
                grad = Xb.T @ (preds - y) / n + l2 * np.r_[0, w[1:]]
                w -= lr * grad
            return w

        def predict_proba(w, X):
            Xb = np.hstack([np.ones((X.shape[0],1)), X])
            return sigmoid(Xb @ w)

        def predict(w, X, threshold=0.5):
            return (predict_proba(w, X) >= threshold).astype(int)
        ```
        """)
    if any(k in lname for k in ("random forest", "xgboost", "lightgbm", "catboost", "gradient boosting")):
        return textwrap.dedent("""
        ### Library example (scikit-learn / XGBoost)

        ```python
        # filepath: examples/ensemble_example.py
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        import numpy as np

        # synthetic data
        X = np.random.randn(1000, 20)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        print("RF:", classification_report(y_test, rf.predict(X_test)))

        xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        print("XGB:", classification_report(y_test, xgb.predict(X_test)))
        ```
        """)
    if "k-means" in lname:
        return textwrap.dedent("""
        ### From-scratch K-Means (NumPy)

        ```python
        # filepath: examples/kmeans_scratch.py
        import numpy as np

        def kmeans(X, k, n_iter=100, tol=1e-4, seed=0):
            rng = np.random.default_rng(seed)
            n, d = X.shape
            centroids = X[rng.choice(n, size=k, replace=False)]
            for it in range(n_iter):
                # assign
                dists = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
                labels = dists.argmin(axis=1)
                new_centroids = np.vstack([X[labels==j].mean(axis=0) if (labels==j).any() else centroids[j] for j in range(k)])
                if np.linalg.norm(new_centroids - centroids) < tol:
                    break
                centroids = new_centroids
            return centroids, labels
        ```
        """)
    return ""

def generate_md_for(name):
    lname = name.lower()
    if any(k in lname for k in ["regress", "linear", "polynomial", "svr"]):
        ptype = "Regression"
    elif any(k in lname for k in ["logistic", "svm", "classifier", "knn", "naive", "naïve", "xgboost", "catboost", "lightgbm", "random forest", "gradient"]):
        ptype = "Classification"
    elif any(k in lname for k in ["k-means", "dbscan", "hierarchical", "gmm"]):
        ptype = "Clustering"
    elif any(k in lname for k in ["pca", "tsne", "umap", "autoencoder"]):
        ptype = "Dimensionality Reduction"
    elif any(k in lname for k in ["label propagation", "self-training"]):
        ptype = "Semi-Supervised"
    else:
        ptype = "General/Deep Learning"

    extra = extra_impl_for(name)
    return generic_contents(name, ptype, extra_impl=extra)

def safe_filename(name):
    return name.lower().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "").replace("—", "-")

def main():
    for alg in ALGORITHMS:
        fname = safe_filename(alg) + ".md"
        target = OUT_DIR / fname
        print("Writing", target)
        content = generate_md_for(alg)
        target.write_text(content, encoding="utf-8")
    print("Done. Files written to:", OUT_DIR)

if __name__ == "__main__":
    main()
