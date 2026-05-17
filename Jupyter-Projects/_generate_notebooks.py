"""One-off generator for 20 Jupyter notebooks. Run once; do not execute notebook cells."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def md(source: str) -> dict:
    if not source.endswith("\n"):
        source += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source: str) -> dict:
    lines = source.strip("\n") + "\n"
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": lines.splitlines(keepends=True)}

def nb(cells, title=""):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

def save(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)

def common_pip():
    return code("""# Optional: install dependencies (uncomment if needed)
# !pip install -q numpy pandas matplotlib seaborn scikit-learn tensorflow requests yfinance

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set_theme(style="whitegrid")
print("TensorFlow:", tf.__version__)
""")

def common_split_classification():
    return code("""# Train / validation / test split (stratified for classification)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
)  # ~70/15/15

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("Train:", X_train_s.shape, "Val:", X_val_s.shape, "Test:", X_test_s.shape)
""")

def common_callbacks(path="best_model.keras"):
    return code(f"""checkpoint = callbacks.ModelCheckpoint(
    "{path}", monitor="val_loss", save_best_only=True, verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
cb_list = [checkpoint, early_stop, reduce_lr]
""")

def common_train_eval():
    return code(f"""history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=80,
    batch_size=32,
    callbacks=cb_list,
    verbose=1,
)

# Loss curves
pd.DataFrame(history.history).plot(figsize=(10, 4))
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.show()

# Test metrics
y_prob = model.predict(X_test_s, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
except Exception:
    pass
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()
""")

def common_inference(model_path="best_model.keras", label="positive class"):
    return code(f"""# Inference demo on a few test rows
loaded = keras.models.load_model("{model_path}")
sample_idx = np.arange(min(5, len(X_test_s)))
samples = X_test_s[sample_idx]
probs = loaded.predict(samples, verbose=0).ravel()
for i, p in zip(sample_idx, probs):
    print(f"Row {{i}} -> P({label}): {{p:.4f}}, predicted: {{int(p >= 0.5)}}, actual: {{int(y_test[i])}}")

import joblib
joblib.dump(scaler, "scaler.pkl")
print("Saved scaler.pkl and {model_path}")
""")

def deploy_notes(domain="classification"):
    return md(f"""## Deployment Notes

1. **Serving**: Export with `model.export("saved_model")` for TensorFlow Serving, or wrap `predict` in FastAPI/Flask.
2. **Preprocessing**: Always apply the **same** scaler/encoder fitted on training data (`scaler.pkl`).
3. **Monitoring**: Track input drift, latency, and {"class balance / fraud rate" if "fraud" in domain else "prediction distribution"} on live traffic.
4. **Retraining**: Schedule periodic retrain when performance drops below SLA.
5. **Security**: Do not log PII; use HTTPS and auth on inference endpoints.
""")

# ---------- ANN NOTEBOOKS ----------

def ann_churn():
    cells = [
        md("""# Retail Customer Churn Prediction with ANN (MLP)

## Objectives
- Predict whether a telecom retail customer will **churn** (leave).
- Build an end-to-end **feedforward neural network** (multi-layer perceptron).

## Theory (Simple English)
An **ANN** stacks layers of neurons. Each neuron computes a weighted sum of inputs plus bias, then applies an **activation** (ReLU for hidden layers, sigmoid for binary output).

**Forward pass:** $z = Wx + b$, $a = \\text{ReLU}(z)$

**Loss (binary classification):** Binary cross-entropy compares predicted probability to true label 0/1.

**Why MLP here?** Tabular features (tenure, charges, contract type) map well to dense layers after encoding categoricals.
"""),
        common_pip(),
        md("""## Problem Definition & Business Context

**Churn** costs retailers and telcos heavily (acquisition > retention).  
**Goal:** Flag high-risk customers early for retention offers.  
**Success metric:** Recall on churners (catch leavers) with acceptable precision.
"""),
        code("""import io
import requests

# Telco Customer Churn — public mirror (IBM sample, stable CSV structure)
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
r = requests.get(URL, timeout=60)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text))
print(df.shape)
df.head()
"""),
        code("""# EDA
print(df.info())
print(df["Churn"].value_counts(normalize=True))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x="Churn", ax=axes[0])
axes[0].set_title("Churn distribution")
if "MonthlyCharges" in df.columns:
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
    axes[1].set_title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Numeric feature correlations")
    plt.show()
"""),
        code("""# Preprocessing
df = df.drop(columns=["customerID"], errors="ignore")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
y = df["Churn"].values
X_df = df.drop(columns=["Churn"])

# One-hot encode categoricals
X_df = pd.get_dummies(X_df, drop_first=True)
X = X_df.values.astype(np.float32)
feature_names = list(X_df.columns)
print("Features:", len(feature_names))
"""),
        common_split_classification(),
        code("""# Build MLP (ANN)
model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)
model.summary()
"""),
        common_callbacks("ann_churn_best.keras"),
        common_train_eval(),
        common_inference("ann_churn_best.keras"),
        deploy_notes(),
    ]
    return cells

def ann_loan():
    cells = [
        md("""# Banking Loan Default Risk with ANN

## Objectives
Predict **credit default risk** using the UCI German Credit dataset with an MLP.

## Theory
**Feedforward ANN** learns non-linear boundaries between credit features and default label.  
**Architecture:** Input (financial attributes) → hidden ReLU → sigmoid (bad credit probability).
"""),
        common_pip(),
        md("## Business Context\nBanks must estimate **probability of default (PD)** for pricing and capital requirements (Basel). False negatives (approving bad loans) are costly."),
        code("""# UCI German Credit (statlog)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
names = [f"A{i}" for i in range(1, 21)] + ["class"]
df = pd.read_csv(URL, sep=r"\\s+", header=None, names=names)
# class: 1 = good, 2 = bad -> default = bad
df["default"] = (df["class"] == 2).astype(int)
df = df.drop(columns=["class"])
print(df.head())
"""),
        code("""# EDA — all features categorical in original encoding
print(df["default"].value_counts(normalize=True))
sns.countplot(data=df, x="default")
plt.title("Default rate (1=bad credit)")
plt.show()
"""),
        code("""# Encode categoricals as integers then one-hot
X_df = pd.get_dummies(df.drop(columns=["default"]), drop_first=True)
y = df["default"].values
X = X_df.values.astype(np.float32)
"""),
        common_split_classification(),
        code("""model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.summary()
"""),
        common_callbacks("ann_loan_best.keras"),
        common_train_eval(),
        code("""loaded = keras.models.load_model("ann_loan_best.keras")
for i in range(min(5, len(X_test_s))):
    p = loaded.predict(X_test_s[i:i+1], verbose=0)[0, 0]
    print(f"Applicant {i}: P(default)={p:.3f}, pred={int(p>=0.5)}, actual={y_test[i]}")
import joblib
joblib.dump(scaler, "loan_scaler.pkl")
"""),
        deploy_notes("banking"),
    ]
    return cells

def ann_attrition():
    cells = [
        md("""# Employee Attrition Prediction with ANN

## Objectives
Predict **employee attrition** (HR analytics) using IBM HR Analytics Employee Attrition dataset.

## Architecture
MLP: dense layers with dropout to reduce overfitting on tabular HR features.
"""),
        common_pip(),
        md("## Business Context\nHigh attrition increases hiring cost. HR uses models to target retention programs."),
        code("""URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
df = pd.read_csv(URL)
print(df.shape)
df.head()
"""),
        code("""print(df["Attrition"].value_counts())
sns.countplot(data=df, x="Attrition")
plt.show()
"""),
        code("""target_col = "Attrition"
df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
drop_cols = [c for c in ["EmployeeNumber", "Over18", "StandardHours"] if c in df.columns]
X_df = df.drop(columns=[target_col] + drop_cols)
X_df = pd.get_dummies(X_df, drop_first=True)
y = df[target_col].values
X = X_df.values.astype(np.float32)
"""),
        common_split_classification(),
        code("""model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(96, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(48, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
model.summary()
"""),
        common_callbacks("ann_attrition_best.keras"),
        common_train_eval(),
        common_inference("ann_attrition_best.keras"),
        deploy_notes(),
    ]
    return cells

def ann_fraud():
    cells = [
        md("""# Credit Card Fraud Detection with ANN (Imbalanced)

## Objectives
Detect fraudulent transactions on highly **imbalanced** data using an MLP.

## Theory
Use **class weights** or careful metrics (precision/recall, PR-AUC). ANN learns patterns in amount, time, PCA features.
"""),
        common_pip(),
        md("## Business Context\nFraud causes direct losses; models must balance **false alarms** vs **missed fraud**."),
        code("""# OpenML credit card fraud (sklearn fetch — downloads from internet)
from sklearn.datasets import fetch_openml

print("Downloading OpenML creditcard dataset (may take a minute)...")
data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
df = data.frame
print(df.shape)
df.head()
"""),
        code("""# EDA — extreme imbalance
if "Class" in df.columns:
    y_col = "Class"
else:
    y_col = df.columns[-1]
df[y_col] = df[y_col].astype(int)
print(df[y_col].value_counts())
sns.countplot(data=df, x=y_col)
plt.title("Fraud (1) vs Normal (0)")
plt.show()
"""),
        code("""# Subsample for notebook runtime (optional — comment out for full data)
# Use stratified sample to keep fraud cases
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, train_size=50000, random_state=SEED)
idx, _ = next(sss.split(df, df[y_col]))
df = df.iloc[idx].reset_index(drop=True)

X = df.drop(columns=[y_col]).values.astype(np.float32)
y = df[y_col].values
"""),
        common_split_classification(),
        code("""# Class weight for imbalance
neg, pos = np.bincount(y_train.astype(int))
total = neg + pos
weight_for_1 = (total / (2.0 * pos)) if pos > 0 else 1.0
class_weight = {0: 1.0, 1: weight_for_1}
print("class_weight:", class_weight)

model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
model.summary()
"""),
        common_callbacks("ann_fraud_best.keras"),
        code("""cb_list  # defined above
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=50,
    batch_size=256,
    class_weight=class_weight,
    callbacks=cb_list,
    verbose=1,
)
y_prob = model.predict(X_test_s, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Fraud detection confusion matrix")
plt.show()
"""),
        common_inference("ann_fraud_best.keras"),
        deploy_notes("fraud"),
    ]
    return cells

def ann_housing():
    cells = [
        md("""# California Housing Price Regression with ANN

## Objectives
Predict **median house value** (regression) with a feedforward network.

## Theory
**Regression output:** single neuron with linear activation.  
**Loss:** Mean Squared Error (MSE) — penalizes large errors quadratically.
"""),
        common_pip(),
        md("## Business Context\nLenders and investors estimate property values; regression ANNs capture non-linear feature interactions."),
        code("""from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
"""),
        code("""sns.histplot(df["MedHouseVal"], kde=True)
plt.title("Target: Median House Value")
plt.show()
sns.pairplot(df[["MedInc", "HouseAge", "MedHouseVal"]], diag_kind="hist", corner=True)
plt.show()
"""),
        code("""X = df.drop(columns=["MedHouseVal"]).values.astype(np.float32)
y = df["MedHouseVal"].values.astype(np.float32)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=SEED)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
"""),
        code("""model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear"),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
"""),
        code("""cb_reg = [
    callbacks.ModelCheckpoint("ann_housing_best.keras", save_best_only=True, monitor="val_loss"),
    callbacks.EarlyStopping(patience=15, restore_best_weights=True),
]
history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=100,
    batch_size=64,
    callbacks=cb_reg,
    verbose=1,
)
pd.DataFrame(history.history)[["loss", "val_loss"]].plot()
plt.show()
"""),
        code("""pred_s = model.predict(X_test_s, verbose=0).ravel()
pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
plt.scatter(y_test, pred, alpha=0.3, s=5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Housing price: actual vs predicted")
plt.show()
"""),
        code("""model.save("ann_housing_final.keras")
import joblib
joblib.dump(scaler, "housing_X_scaler.pkl")
joblib.dump(y_scaler, "housing_y_scaler.pkl")
"""),
        deploy_notes("regression"),
    ]
    return cells

# ---------- CNN NOTEBOOKS ----------

def cnn_template(title, dataset_loader, num_classes, task_md):
    cells = [
        md(f"""# {title}

## Objectives
{task_md}

## CNN Theory (Simple English)
**Convolution** slides small filters (kernels) over the image to detect edges, textures, and shapes.  
**Pooling** downsamples spatial size.  
**Stack:** Conv → ReLU → Pool → ... → Flatten → Dense → Softmax (multi-class).

**Formula (2D conv output size):**  
$O = \\lfloor (I - K + 2P) / S \\rfloor + 1$ where $I$=input size, $K$=kernel, $P$=padding, $S$=stride.
"""),
        common_pip(),
        md("## Problem Definition\nImage classification assigns a label to each image. Used in OCR, quality control, medical screening."),
        code(dataset_loader),
        code(f"""# EDA — sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, (img, label) in zip(axes.ravel(), train_ds.take(10)):
    arr = img.numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
    ax.set_title(str(int(label.numpy() if hasattr(label, 'numpy') else label)))
    ax.axis("off")
plt.suptitle("Sample training images")
plt.show()
"""),
        code("""# Normalize and batch
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_batched = train_ds.map(preprocess).shuffle(10000).batch(64).prefetch(AUTOTUNE)
val_batched = val_ds.map(preprocess).batch(64).prefetch(AUTOTUNE)
test_batched = test_ds.map(preprocess).batch(64).prefetch(AUTOTUNE)
"""),
        code(f"""model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense({num_classes}, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
"""),
        code("""cnn_cb = [
    callbacks.ModelCheckpoint("cnn_best.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy", mode="max"),
]
history = model.fit(train_batched, validation_data=val_batched, epochs=25, callbacks=cnn_cb, verbose=1)
"""),
        code("""test_loss, test_acc = model.evaluate(test_batched, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Plot history
pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot(figsize=(8,4))
plt.title("CNN training accuracy")
plt.show()

y_true, y_pred = [], []
for images, labels in test_batched:
    preds = model.predict(images, verbose=0).argmax(axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())
print(classification_report(y_true, y_pred))
"""),
        code("""# Inference on one batch
for images, labels in test_batched.take(1):
    probs = model.predict(images[:3], verbose=0)
    for i in range(3):
        print("True:", int(labels[i]), "Pred:", probs[i].argmax(), "Confidence:", probs[i].max():.3f)
model.save("cnn_deployed.keras")
"""),
        deploy_notes("CNN images"),
    ]
    return cells

def cnn_mnist():
    loader = """(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
input_shape = (28, 28, 1)
num_classes = 10

# Use part of train as validation
val_size = 10000
train_ds = tf.data.Dataset.from_tensor_slices((x_train[val_size:], y_train[val_size:]))
val_ds = tf.data.Dataset.from_tensor_slices((x_train[:val_size], y_train[:val_size]))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
"""
    cells = cnn_template("MNIST Handwritten Digit Classification with CNN", loader, 10,
        "Classify 28×28 grayscale digits (0–9) using a convolutional neural network.")
    return cells

def cnn_fashion():
    loader = """(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
input_shape = (28, 28, 1)
num_classes = 10
class_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

val_size = 10000
train_ds = tf.data.Dataset.from_tensor_slices((x_train[val_size:], y_train[val_size:]))
val_ds = tf.data.Dataset.from_tensor_slices((x_train[:val_size], y_train[:val_size]))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
"""
    return cnn_template("Fashion-MNIST Apparel Classification with CNN", loader, 10,
        "Classify clothing items from Fashion-MNIST.")

def cnn_cifar():
    loader = """(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.squeeze(), y_test.squeeze()
input_shape = (32, 32, 3)
num_classes = 10
val_size = 5000
train_ds = tf.data.Dataset.from_tensor_slices((x_train[val_size:], y_train[val_size:]))
val_ds = tf.data.Dataset.from_tensor_slices((x_train[:val_size], y_train[:val_size]))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
"""
    return cnn_template("CIFAR-10 Object Classification with CNN", loader, 10,
        "Classify 32×32 color images into 10 object categories.")

def cnn_xray():
    cells = [
        md("""# Chest X-Ray Pneumonia Detection with CNN (Transfer Learning)

## Objectives
Binary image classification: **Normal** vs **Pneumonia** using CNN + MobileNetV2 backbone.

## CNN + Transfer Learning
**Transfer learning** reuses weights trained on ImageNet. We freeze early layers and train only the top **classifier head**, saving data and training time.

## Note
Download the [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and extract to `chest_xray_data/`. A fallback demo uses CIFAR10 if folders are missing.
"""),
        common_pip(),
        md("""## Problem Definition & Business Context
Hospitals use imaging AI to **triage** chest X-rays. Models assist radiologists; they require clinical validation, bias audits, and regulatory approval before production use.
"""),
        code("""import zipfile
import requests
from pathlib import Path

# Small public sample — user may replace with full Kaggle download
# Using tensorflow_demo style: we build from directory structure after manual download OR use cats/dogs style subset
# Stable approach: use keras TSV + folder from TFDS alternative — here we use a documented folder layout

DATA_DIR = Path("chest_xray_data")
if not (DATA_DIR / "train").exists():
    print("Download chest x-ray zip from Kaggle and extract to chest_xray_data/train and test")
    print("Expected: chest_xray_data/train/NORMAL, chest_xray_data/train/PNEUMONIA, ...")
else:
    print("Found local chest xray folders")

IMG_SIZE = (160, 160)
BATCH = 32

if (DATA_DIR / "train").exists():
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, validation_split=0.2, subset="training"
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, validation_split=0.2, subset="validation"
    )
    test_ds = keras.utils.image_dataset_from_directory(DATA_DIR / "test", image_size=IMG_SIZE, batch_size=BATCH)
    class_names = train_ds.class_names
    print("Classes:", class_names)
else:
    # Fallback demo pipeline with CIFAR grayscale simulation (documented)
    print("Using CIFAR10 subset as DEMO pipeline when chest data missing — replace with real x-ray folders.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    binary = (y_train < 2).astype(int), (y_test < 2).astype(int)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train[:5000], binary[0][:5000])).batch(BATCH)
    val_ds = tf.data.Dataset.from_tensor_slices((x_train[5000:6000], binary[0][5000:6000])).batch(BATCH)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test[:1000], binary[1][:1000])).batch(BATCH)
"""),
        code("""# EDA — class balance and sample images
if 'class_names' in dir():
    counts = {c: 0 for c in class_names}
    for _, labels in train_ds:
        for lab in labels.numpy():
            counts[class_names[int(lab)]] += 1
        break
    print("Sample batch class counts:", counts)

for images, labels in train_ds.take(1):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, lab in zip(axes.ravel(), images[:8], labels[:8]):
        ax.imshow((img.numpy() * 255).astype("uint8") if img.numpy().max() <= 1 else img.numpy().astype("uint8"))
        ax.set_title(class_names[int(lab)] if 'class_names' in dir() else str(int(lab)))
        ax.axis("off")
    plt.suptitle("Sample X-ray / demo images")
    plt.show()
"""),
        code("""AUTOTUNE = tf.data.AUTOTUNE

def prep(img, label):
    img = tf.image.resize(img, (160, 160))
    img = tf.cast(img, tf.float32)
    return keras.applications.mobilenet_v2.preprocess_input(img), label

train_batched = train_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_batched = val_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_batched = test_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
"""),
        code("""base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(160, 160, 3))
base.trainable = False
inputs = layers.Input(shape=(160, 160, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.summary()
"""),
        code("""xray_cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
    callbacks.ModelCheckpoint("cnn_xray_best.keras", save_best_only=True, monitor="val_auc", mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]
history = model.fit(train_batched, validation_data=val_batched, epochs=15, callbacks=xray_cb, verbose=1)
pd.DataFrame(history.history).plot(figsize=(10,4))
plt.title("Transfer learning training curves")
plt.show()
"""),
        code("""loss, acc, auc = model.evaluate(test_batched, verbose=0)
print(f"Test loss={loss:.4f} acc={acc:.4f} auc={auc:.4f}")

y_true, y_prob = [], []
for imgs, labs in test_batched:
    p = model.predict(imgs, verbose=0).ravel()
    y_prob.extend(p)
    y_true.extend(labs.numpy())
y_pred = (np.array(y_prob) >= 0.5).astype(int)
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Pneumonia detection — confusion matrix")
plt.show()
"""),
        code("""# Inference demo
for imgs, labs in test_batched.take(1):
    p = model.predict(imgs[:3], verbose=0).ravel()
    for i, prob in enumerate(p):
        print(f"Image {i}: P(pneumonia)={prob:.3f} pred={int(prob>=0.5)} true={int(labs[i])}")
model.save("cnn_xray_deploy.keras")
"""),
        deploy_notes("medical imaging — requires clinical validation"),
    ]
    return cells

def cnn_plant():
    loader = """# Plant pathology — use tensorflow flowers as stable proxy for leaf disease CNN practice
# tf.keras.utils.get_file can fetch flower photos
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
import tarfile, urllib.request
data_root = pathlib.Path(keras.utils.get_file('flower_photos', origin=dataset_url, extract=True))
flower_root = list(data_root.glob('*'))[0] if data_root.name != 'flower_photos' else data_root
if not (flower_root / 'daisy').exists():
    flower_root = data_root / 'flower_photos'

IMG_SIZE = (180, 180)
BATCH = 32
train_ds = keras.utils.image_dataset_from_directory(flower_root, validation_split=0.2, subset='training', seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = keras.utils.image_dataset_from_directory(flower_root, validation_split=0.2, subset='validation', seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
test_ds = val_ds  # use val as holdout for notebook
num_classes = len(train_ds.class_names)
input_shape = IMG_SIZE + (3,)
class_names = train_ds.class_names
print(class_names)
"""
    cells = [
        md("""# Plant / Flower Image Classification with CNN

## Objectives
Multi-class **leaf/flower** image classification (TensorFlow flower photos as stable public dataset).

## CNN Architecture
Depthwise feature extraction with Conv2D blocks + dense classifier head.
"""),
        common_pip(),
        md("## Business Context\nAgriculture and retail garden centers use vision models for disease detection and inventory."),
        code(loader),
        code("""# EDA
plt.figure(figsize=(10,4))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i+1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(class_names[labels[i]])
        ax.axis("off")
plt.show()
"""),
        code("""normalization = layers.Rescaling(1./255)
train_scaled = train_ds.map(lambda x,y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
val_scaled = val_ds.map(lambda x,y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
"""),
        code("""model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
"""),
        code("""history = model.fit(train_scaled, validation_data=val_scaled, epochs=20, callbacks=[
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('cnn_plant_best.keras', save_best_only=True),
])
pd.DataFrame(history.history).plot()
plt.show()
"""),
        code("""model.evaluate(val_scaled, verbose=0)
model.save('cnn_plant_final.keras')
for img, lbl in train_scaled.take(1):
    pred = model.predict(img[:2], verbose=0).argmax(axis=1)
    print('Labels:', lbl[:2].numpy(), 'Pred:', pred)
"""),
        deploy_notes("CNN agriculture"),
    ]
    return cells

# ---------- RNN NOTEBOOKS ----------

def rnn_imdb():
    cells = [
        md("""# IMDB Movie Review Sentiment with LSTM

## Objectives
Binary **sentiment classification** (positive/negative) on text using LSTM.

## RNN Theory
**RNN** maintains a hidden state across time steps: $h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$  
**LSTM** adds gates (forget, input, output) to capture long-range dependencies and mitigate vanishing gradients.
"""),
        common_pip(),
        md("## Business Context\nRetailers and banks analyze reviews, complaints, and chat logs for sentiment routing."),
        code("""max_features = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=SEED)
"""),
        code("""# EDA — sequence lengths
train_lens = [len(s) for s in x_train[:5000]]
sns.histplot(train_lens, bins=30)
plt.title("Review sequence lengths (subset)")
plt.xlabel("Tokens")
plt.show()
"""),
        code("""# Pad sequences to fixed length
x_train_p = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_val_p = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_len)
x_test_p = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
"""),
        code("""model = models.Sequential([
    layers.Embedding(max_features, 128, input_length=max_len),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.summary()
"""),
        code("""rnn_cb = [
    callbacks.ModelCheckpoint("lstm_imdb_best.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max"),
]
history = model.fit(
    x_train_p, y_train,
    validation_data=(x_val_p, y_val),
    epochs=5,
    batch_size=128,
    callbacks=rnn_cb,
)
"""),
        code("""test_loss, test_acc, test_auc = model.evaluate(x_test_p, y_test, verbose=0)
print(f"Test acc: {test_acc:.4f}")
y_prob = model.predict(x_test_p[:500], verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_test[:500], y_pred))
"""),
        code("""# Inference demo
sample = x_test_p[:3]
probs = model.predict(sample, verbose=0)
for i, p in enumerate(probs):
    print(f"Review {i}: sentiment={'positive' if p[0]>0.5 else 'negative'} ({p[0]:.3f})")
model.save("lstm_imdb_final.keras")
"""),
        deploy_notes("NLP sentiment"),
    ]
    return cells

def rnn_stock():
    cells = [
        md("""# Stock Price Forecasting with LSTM

## Objectives
Forecast next-day **closing price** using past window of prices (univariate time series).

## Theory
LSTM remembers patterns over **sequences** of timesteps; we slide a window of `lookback` days to predict day $t+1$.
"""),
        common_pip(),
        md("## Business Context\nBanks and funds use forecasts for risk; retail investors for planning (not financial advice)."),
        code("""import yfinance as yf

ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-12-31", progress=False)
df = df[["Close"]].dropna()
print(df.tail())
"""),
        code("""sns.lineplot(data=df["Close"])
plt.title(f"{ticker} closing price")
plt.show()
"""),
        code("""lookback = 60
data = df["Close"].values.astype(np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

def make_sequences(series, lb):
    X, y = [], []
    for i in range(lb, len(series)):
        X.append(series[i-lb:i])
        y.append(series[i])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, lookback)
X = X[..., np.newaxis]  # (samples, timesteps, features)

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
"""),
        code("""model = models.Sequential([
    layers.Input(shape=(lookback, 1)),
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
"""),
        code("""history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint("lstm_stock_best.keras", save_best_only=True),
    ],
    verbose=1,
)
"""),
        code("""pred_scaled = model.predict(X_test, verbose=0).ravel()
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
print("RMSE:", np.sqrt(mean_squared_error(actual, pred)))
plt.plot(actual, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Stock close price forecast (test)")
plt.show()
"""),
        code("""import joblib
joblib.dump(scaler, "stock_scaler.pkl")
model.save("lstm_stock_final.keras")
"""),
        deploy_notes("time series — not investment advice"),
    ]
    return cells

def rnn_energy():
    cells = [
        md("""# Energy Consumption Forecasting with LSTM

## Objectives
Forecast **household power consumption** (daily mean kW) using LSTM.

## RNN Theory
Time-series forecasting feeds a **sequence of past days** into LSTM; the network outputs the next day's consumption. Gates help remember weekly cycles (weekends vs weekdays).

## Data
UCI Individual Household Electric Power Consumption (aggregated daily).
"""),
        common_pip(),
        md("## Business Context\nUtilities and smart-home apps forecast load for **grid planning**, pricing, and anomaly detection."),
        code("""URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
import zipfile, io, requests
r = requests.get(URL, timeout=120)
z = zipfile.ZipFile(io.BytesIO(r.content))
fname = [n for n in z.namelist() if n.endswith(".txt")][0]
with z.open(fname) as f:
    raw = pd.read_csv(f, sep=";", low_memory=False, na_values=["?"])
raw["datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], dayfirst=True)
raw = raw.set_index("datetime").sort_index()
# Daily mean global active power
daily = raw["Global_active_power"].astype(float).resample("D").mean().dropna().to_frame()
print(daily.head())
"""),
        code("""daily.plot(figsize=(12,3))
plt.title("Daily mean global active power (kW)")
plt.show()
"""),
        code("""values = daily.values.astype(np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values).flatten()
lookback = 30

def make_seq(s, lb):
    X, y = [], []
    for i in range(lb, len(s)):
        X.append(s[i-lb:i])
        y.append(s[i])
    return np.array(X), np.array(y)

X, y = make_seq(scaled, lookback)
X = X[..., np.newaxis]
n = len(X)
tr, va = int(0.7*n), int(0.85*n)
X_train, y_train = X[:tr], y[:tr]
X_val, y_val = X[tr:va], y[tr:va]
X_test, y_test = X[va:], y[va:]
"""),
        code("""model = models.Sequential([
    layers.LSTM(64, input_shape=(lookback, 1)),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
"""),
        code("""energy_cb = [
    callbacks.ModelCheckpoint("lstm_energy_best.keras", save_best_only=True),
    callbacks.EarlyStopping(patience=6, restore_best_weights=True),
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30,
          callbacks=energy_cb, verbose=1, batch_size=32)
pd.DataFrame(history.history).plot()
plt.show()
"""),
        code("""pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
actual = scaler.inverse_transform(y_test.reshape(-1,1))
print("RMSE:", np.sqrt(mean_squared_error(actual, pred)))
plt.plot(actual, label="actual")
plt.plot(pred, label="pred")
plt.legend()
plt.title("Energy consumption — test forecast")
plt.show()
"""),
        code("""# Inference — next day forecast
w = scaled[-lookback:].reshape(1, lookback, 1)
nxt = scaler.inverse_transform(model.predict(w, verbose=0))[0,0]
print(f"Next-day power forecast (kW): {nxt:.4f}")
model.save("lstm_energy.keras")
import joblib
joblib.dump(scaler, "energy_scaler.pkl")
"""),
        deploy_notes("energy forecasting"),
    ]
    return cells

def rnn_air():
    cells = [
        md("""# Air Passengers Time Series Forecasting with Simple RNN

## Objectives
Classic **Air Passengers** monthly series — forecast with a vanilla **SimpleRNN**.

## Theory
Vanilla **RNN** is simpler than LSTM: $h_t = \\tanh(W x_t + U h_{t-1} + b)$. It works on smooth seasonal series but may struggle with very long dependencies compared to LSTM/GRU.
"""),
        common_pip(),
        md("## Business Context\nAirlines and tourism use passenger forecasts for **capacity planning** (routes, staff, fuel)."),
        code("""# Built-in statsmodels dataset via pandas URL
url = "https://raw.githubusercontent.com/plotly/datasets/master/air-passengers.csv"
df = pd.read_csv(url)
df.columns = ["Month", "Passengers"]
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month")
print(df.head())
"""),
        code("""df.plot(title="Air Passengers")
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(df["Passengers"], model="multiplicative", period=12)
decomp.plot()
plt.show()
"""),
        code("""series = df["Passengers"].values.astype(np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
lookback = 12

def make_seq(s, lb):
    X, y = [], []
    for i in range(lb, len(s)):
        X.append(s[i-lb:i])
        y.append(s[i])
    return np.array(X), np.array(y)

X, y = make_seq(scaled, lookback)
X = X[..., np.newaxis]
split = int(len(X)*0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
val_idx = int(0.85 * len(X_train))
X_tr, y_tr = X_train[:val_idx], y_train[:val_idx]
X_val, y_val = X_train[val_idx:], y_train[val_idx:]
"""),
        code("""model = models.Sequential([
    layers.SimpleRNN(32, input_shape=(lookback, 1)),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
"""),
        code("""air_cb = [
    callbacks.ModelCheckpoint("rnn_air_best.keras", save_best_only=True),
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
]
history = model.fit(X_tr, y_tr, epochs=80, batch_size=8, verbose=1,
          validation_data=(X_val, y_val), callbacks=air_cb)
pd.DataFrame(history.history)["loss"].plot()
plt.title("Air Passengers RNN loss")
plt.show()
"""),
        code("""pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
actual = scaler.inverse_transform(y_test.reshape(-1,1))
print("RMSE:", np.sqrt(mean_squared_error(actual, pred)))
plt.plot(actual, label="Actual")
plt.plot(pred, label="RNN Pred")
plt.legend()
plt.show()
"""),
        code("""# Inference — next month passengers
w = scaled[-lookback:].reshape(1, lookback, 1)
nxt = scaler.inverse_transform(model.predict(w, verbose=0))[0,0]
print(f"Next-month passenger forecast: {nxt:.0f}")
model.save("rnn_air_passengers.keras")
"""),
        deploy_notes(),
    ]
    return cells

def rnn_char():
    cells = [
        md("""# Character-Level Text Generation with RNN

## Objectives
Train a **char-RNN** to generate Shakespeare-like text character by character.

## Theory
Each character is encoded; the LSTM predicts the **next character** via softmax over the vocabulary. **Temperature** scales randomness at generation time (higher = more creative, lower = more conservative).
"""),
        common_pip(),
        md("## Business Context\nGenerative RNNs underpin autocomplete, marketing copy drafts, and chatbots (modern systems use Transformers, but char-RNNs teach sequence fundamentals)."),
        code("""import requests
url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
text = requests.get(url, timeout=30).text
print("Characters:", len(text))
print(text[:200])
"""),
        code("""chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = np.array(chars)
vocab_size = len(chars)

seq_length = 100
step = 3
sequences = []
next_chars = []
for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i+seq_length])
    next_chars.append(text[i+seq_length])

x = np.array([[char2idx[c] for c in seq] for seq in sequences])
y = np.array([char2idx[c] for c in next_chars])
print(x.shape, y.shape)
"""),
        code("""# One-hot encode for training (subset for speed)
subset = 20000
x_ohe = keras.utils.to_categorical(x[:subset], num_classes=vocab_size)
y_ohe = keras.utils.to_categorical(y[:subset], num_classes=vocab_size)
x_train, x_val, y_train, y_val = train_test_split(x_ohe, y_ohe, test_size=0.1, random_state=SEED)
print("Train sequences:", x_train.shape)
"""),
        code("""model = models.Sequential([
    layers.LSTM(128, input_shape=(seq_length, vocab_size)),
    layers.Dense(vocab_size, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
"""),
        code("""char_cb = [
    callbacks.ModelCheckpoint("char_rnn_best.keras", save_best_only=True),
    callbacks.EarlyStopping(patience=3, restore_best_weights=True),
]
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64,
          callbacks=char_cb, verbose=1)
pd.DataFrame(history.history).plot()
plt.show()
"""),
        code("""def generate(seed, length=200, temperature=1.0):
    out = seed
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, vocab_size))
        seq = out[-seq_length:]
        for t, ch in enumerate(seq):
            x_pred[0, t, char2idx[ch]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        idx = np.random.choice(len(preds), p=exp_preds / np.sum(exp_preds))
        out += idx2char[idx]
    return out

print(generate("ROMEO: ", length=300))
model.save("char_rnn_shakespeare.keras")
"""),
        deploy_notes("generative text"),
    ]
    return cells

# ---------- RETAIL / BANKING ----------

def rb_online_retail():
    cells = [
        md("""# Online Retail Customer Segmentation (Retail Domain)

## Objectives
**RFM analysis** + ANN classifier for high-value segment prediction using UCI Online Retail.

## Business Context
Retailers segment customers for campaigns, loyalty tiers, and churn prevention.
"""),
        common_pip(),
        code("""URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
# Fallback CSV mirror if xlsx fails
try:
    df = pd.read_excel(URL)
except Exception:
    URL2 = "https://raw.githubusercontent.com/plotly/datasets/master/online-retail.csv"
    df = pd.read_csv(URL2)
print(df.shape)
df.head()
"""),
        code("""df = df.dropna(subset=["CustomerID", "InvoiceDate"])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Total"] = df["Quantity"] * df["UnitPrice"]
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Total", "sum"),
)
rfm.head()
"""),
        code("""sns.pairplot(rfm.reset_index(drop=True)[["Recency","Frequency","Monetary"]].sample(500, random_state=SEED))
plt.show()
# High value = top 20% monetary
rfm["HighValue"] = (rfm["Monetary"] >= rfm["Monetary"].quantile(0.8)).astype(int)
"""),
        code("""X = rfm[["Recency","Frequency","Monetary"]].values.astype(np.float32)
y = rfm["HighValue"].values
"""),
        common_split_classification(),
        code("""model = models.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
model.summary()
"""),
        common_callbacks("retail_rfm_best.keras"),
        common_train_eval(),
        common_inference("retail_rfm_best.keras", "high value"),
        deploy_notes("retail CRM"),
    ]
    return cells

def rb_store_sales():
    cells = [
        md("""# Store Sales Forecasting (Retail) with LSTM

## Objectives
Forecast **daily sales** using an LSTM on a public time-series CSV (demand proxy).

## Theory
Retail **demand forecasting** uses past sales windows to predict future units/revenue. LSTMs capture weekly seasonality and trends better than plain moving averages.

**Loss:** MSE on scaled target. **Metric:** RMSE / MAE in original sales units after inverse transform.
"""),
        common_pip(),
        md("""## Problem Definition & Business Context
Stores optimize **inventory**, staffing, and promotions from demand forecasts. Under-forecasting causes stockouts; over-forecasting wastes capital.
"""),
        code("""# Store sales — Kaggle Rossmann requires login; use retail sales time series sample
url = "https://raw.githubusercontent.com/plotly/datasets/master/time-series-19-covid-combined.csv"
df = pd.read_csv(url)
# Use US daily cases as proxy univariate sales-like series for teaching
if "US" in df.columns:
    ts = df[["Date","US"]].dropna().rename(columns={"Date":"date","US":"sales"})
else:
    ts = df.iloc[:, :2]
    ts.columns = ["date","sales"]
ts["date"] = pd.to_datetime(ts["date"])
ts = ts.set_index("date").sort_index()
ts.plot(figsize=(12,3))
plt.title("Retail demand proxy series")
plt.show()

# EDA — rolling mean
ts["roll7"] = ts["sales"].rolling(7).mean()
ts[["sales","roll7"]].plot(figsize=(12,3))
plt.title("Sales vs 7-day rolling mean")
plt.show()
"""),
        code("""values = ts["sales"].values.astype(np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values.reshape(-1,1)).flatten()
lookback = 14

def make_seq(s, lb):
    X, y = [], []
    for i in range(lb, len(s)):
        X.append(s[i-lb:i])
        y.append(s[i])
    return np.array(X), np.array(y)

X, y = make_seq(scaled, lookback)
X = X[..., np.newaxis]
split = int(0.8*len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
val_split = int(0.9 * len(X_train))
X_tr, y_tr = X_train[:val_split], y_train[:val_split]
X_val, y_val = X_train[val_split:], y_train[val_split:]
print("Train", X_tr.shape, "Val", X_val.shape, "Test", X_test.shape)
"""),
        code("""model = models.Sequential([
    layers.LSTM(48, input_shape=(lookback,1)),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
"""),
        code("""store_cb = [
    callbacks.ModelCheckpoint("retail_store_lstm_best.keras", save_best_only=True),
    callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
]
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=40, batch_size=32,
                    callbacks=store_cb, verbose=1)
pd.DataFrame(history.history)[["loss","val_loss"]].plot()
plt.show()
"""),
        code("""pred_scaled = model.predict(X_test, verbose=0)
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)
print(f"Test RMSE: {rmse:.2f} MAE: {mae:.2f}")
plt.figure(figsize=(12,4))
plt.plot(actual, label="actual")
plt.plot(pred, label="forecast")
plt.legend()
plt.title("Retail store sales — test forecast")
plt.show()
"""),
        code("""# Inference: forecast next day from last window
last_window = scaled[-lookback:].reshape(1, lookback, 1)
next_scaled = model.predict(last_window, verbose=0)[0,0]
next_sale = scaler.inverse_transform([[next_scaled]])[0,0]
print(f"Next-day sales forecast: {next_sale:.2f}")
model.save("retail_store_lstm.keras")
import joblib
joblib.dump(scaler, "retail_sales_scaler.pkl")
"""),
        deploy_notes("retail demand"),
    ]
    return cells

def rb_german_credit():
    cells = ann_loan()  # banking focused — duplicate structure ok with different path
    cells[0] = md("""# Banking Credit Risk — German Credit (Retail-Banking Domain)

End-to-end **credit scoring** for a bank using UCI German Credit data and ANN.
Same pipeline as loan default with emphasis on **approve/deny** policy and fair lending checks.
""")
    return cells

def rb_fraud_banking():
    cells = ann_fraud()
    cells[0] = md("""# Banking: Credit Card Fraud Detection (Production-style Pipeline)

Highly imbalanced **transaction fraud** detection for issuers using OpenML credit card data and weighted ANN.
""")
    return cells

def rb_bank_marketing():
    cells = [
        md("""# Bank Marketing — Term Deposit Subscription Prediction

## Objectives
Predict if client subscribes to a **term deposit** (UCI Bank Marketing dataset).

## Business Context
Banks optimize call-center campaigns; uplift modeling reduces wasted contacts.
"""),
        common_pip(),
        code("""URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
df = pd.read_csv(URL, sep=";")
print(df.shape)
print(df["y"].value_counts())
df.head()
"""),
        code("""sns.countplot(data=df, x="y")
plt.title("Term deposit subscription")
plt.show()
# Duration leakage note: remove duration for realistic deployment
if "duration" in df.columns:
    df_model = df.drop(columns=["duration"])
else:
    df_model = df.copy()
"""),
        code("""df_model["y"] = df_model["y"].map({"yes":1, "no":0})
X_df = pd.get_dummies(df_model.drop(columns=["y"]), drop_first=True)
y = df_model["y"].values
X = X_df.values.astype(np.float32)
"""),
        common_split_classification(),
        code("""model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
model.summary()
"""),
        common_callbacks("bank_marketing_best.keras"),
        common_train_eval(),
        common_inference("bank_marketing_best.keras"),
        deploy_notes("bank marketing campaign"),
    ]
    return cells

NOTEBOOKS = [
    (ROOT / "ANN" / "01-Retail-Customer-Churn-ANN.ipynb", ann_churn()),
    (ROOT / "ANN" / "02-Banking-Loan-Default-ANN.ipynb", ann_loan()),
    (ROOT / "ANN" / "03-Employee-Attrition-ANN.ipynb", ann_attrition()),
    (ROOT / "ANN" / "04-Credit-Card-Fraud-ANN.ipynb", ann_fraud()),
    (ROOT / "ANN" / "05-Housing-Price-Regression-ANN.ipynb", ann_housing()),
    (ROOT / "CNN" / "01-MNIST-Digit-Classification-CNN.ipynb", cnn_mnist()),
    (ROOT / "CNN" / "02-Fashion-MNIST-Apparel-CNN.ipynb", cnn_fashion()),
    (ROOT / "CNN" / "03-CIFAR10-Object-Classification-CNN.ipynb", cnn_cifar()),
    (ROOT / "CNN" / "04-Chest-XRay-Pneumonia-CNN.ipynb", cnn_xray()),
    (ROOT / "CNN" / "05-Plant-Flower-Classification-CNN.ipynb", cnn_plant()),
    (ROOT / "RNN" / "01-IMDB-Sentiment-LSTM.ipynb", rnn_imdb()),
    (ROOT / "RNN" / "02-Stock-Price-Forecasting-LSTM.ipynb", rnn_stock()),
    (ROOT / "RNN" / "03-Energy-Consumption-Forecasting-LSTM.ipynb", rnn_energy()),
    (ROOT / "RNN" / "04-Air-Passengers-Forecasting-RNN.ipynb", rnn_air()),
    (ROOT / "RNN" / "05-Shakespeare-Char-RNN.ipynb", rnn_char()),
    (ROOT / "Retail-Banking" / "01-Online-Retail-Customer-Segmentation.ipynb", rb_online_retail()),
    (ROOT / "Retail-Banking" / "02-Store-Sales-Forecasting-Retail.ipynb", rb_store_sales()),
    (ROOT / "Retail-Banking" / "03-Banking-Credit-Risk-German-Credit.ipynb", rb_german_credit()),
    (ROOT / "Retail-Banking" / "04-Credit-Card-Fraud-Detection-Banking.ipynb", rb_fraud_banking()),
    (ROOT / "Retail-Banking" / "05-Bank-Marketing-Term-Deposit.ipynb", rb_bank_marketing()),
]

README_ENTRIES = [
    ("ANN/01-Retail-Customer-Churn-ANN.ipynb", "Telco retail churn prediction with MLP"),
    ("ANN/02-Banking-Loan-Default-ANN.ipynb", "German credit loan default risk (UCI)"),
    ("ANN/03-Employee-Attrition-ANN.ipynb", "HR employee attrition classification"),
    ("ANN/04-Credit-Card-Fraud-ANN.ipynb", "Imbalanced fraud detection (OpenML)"),
    ("ANN/05-Housing-Price-Regression-ANN.ipynb", "California housing price regression"),
    ("CNN/01-MNIST-Digit-Classification-CNN.ipynb", "MNIST digits with ConvNet"),
    ("CNN/02-Fashion-MNIST-Apparel-CNN.ipynb", "Fashion-MNIST clothing CNN"),
    ("CNN/03-CIFAR10-Object-Classification-CNN.ipynb", "CIFAR-10 object CNN"),
    ("CNN/04-Chest-XRay-Pneumonia-CNN.ipynb", "Chest X-ray pneumonia (transfer learning)"),
    ("CNN/05-Plant-Flower-Classification-CNN.ipynb", "Flower/leaf image CNN (TF flowers)"),
    ("RNN/01-IMDB-Sentiment-LSTM.ipynb", "IMDB review sentiment LSTM"),
    ("RNN/02-Stock-Price-Forecasting-LSTM.ipynb", "AAPL stock forecasting LSTM"),
    ("RNN/03-Energy-Consumption-Forecasting-LSTM.ipynb", "Household power LSTM (UCI)"),
    ("RNN/04-Air-Passengers-Forecasting-RNN.ipynb", "Air Passengers classic RNN"),
    ("RNN/05-Shakespeare-Char-RNN.ipynb", "Character-level Shakespeare RNN"),
    ("Retail-Banking/01-Online-Retail-Customer-Segmentation.ipynb", "RFM + ANN high-value segments"),
    ("Retail-Banking/02-Store-Sales-Forecasting-Retail.ipynb", "Retail demand LSTM forecast"),
    ("Retail-Banking/03-Banking-Credit-Risk-German-Credit.ipynb", "Bank credit risk scoring"),
    ("Retail-Banking/04-Credit-Card-Fraud-Detection-Banking.ipynb", "Banking fraud ANN pipeline"),
    ("Retail-Banking/05-Bank-Marketing-Term-Deposit.ipynb", "Term deposit campaign prediction"),
]

def write_readme():
    lines = [
        "# Jupyter Projects — Gen AI & Agentic AI Notes\n",
        "\nTwenty end-to-end notebooks using **TensorFlow/Keras**, with theory markdown and runnable code.\n",
        "**Note:** Cells are not pre-executed. Run locally after `pip install` dependencies.\n",
        "\n## Setup\n",
        "```bash\npip install numpy pandas matplotlib seaborn scikit-learn tensorflow requests yfinance joblib openpyxl statsmodels\n```\n",
        "\n## Notebooks\n",
        "\n### ANN (Feedforward / MLP)\n",
    ]
    for path, desc in README_ENTRIES[:5]:
        lines.append(f"- [{path}]({path}) — {desc}\n")
    lines.append("\n### CNN\n")
    for path, desc in README_ENTRIES[5:10]:
        lines.append(f"- [{path}]({path}) — {desc}\n")
    lines.append("\n### RNN / LSTM\n")
    for path, desc in README_ENTRIES[10:15]:
        lines.append(f"- [{path}]({path}) — {desc}\n")
    lines.append("\n### Retail & Banking\n")
    for path, desc in README_ENTRIES[15:]:
        lines.append(f"- [{path}]({path}) — {desc}\n")
    lines.append("\n## Folder Layout\n")
    lines.append("```\nJupyter-Projects/\n  ANN/          (5 notebooks)\n  CNN/          (5 notebooks)\n  RNN/          (5 notebooks)\n  Retail-Banking/ (5 notebooks)\n```\n")
    (ROOT / "README.md").write_text("".join(lines), encoding="utf-8")

if __name__ == "__main__":
    for path, cells in NOTEBOOKS:
        save(path, cells)
        print("Wrote", path)
    write_readme()
    print("README written")
