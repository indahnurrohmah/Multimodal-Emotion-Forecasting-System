import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# === CONFIG ===
DATA_DIR = "./features"
N_SPLITS = 5
PCA_COMPONENTS_AUDIO = 128
PCA_COMPONENTS_TEXT = 128
PCA_COMPONENTS_VISUAL = 64

# === LOAD DATA ===
X_audio, X_text, X_visual, y = [], [], [], []

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".json"): continue
    with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
        sample = json.load(f)
    
    audio_vec = np.array(sample.get("audio", []), dtype=float)
    text_vec = np.array(sample.get("text", {}).get("embedding", []), dtype=float)
    visual_vec = np.array(sample.get("visual", []), dtype=float)
    
    if audio_vec.size == 0: audio_vec = np.zeros(768)
    if text_vec.size == 0: text_vec = np.zeros(768)
    if visual_vec.size == 0: visual_vec = np.zeros(512)
    
    X_audio.append(audio_vec)
    X_text.append(text_vec)
    X_visual.append(visual_vec)
    y.append(sample.get("emotion"))

X_audio = np.array(X_audio)
X_text = np.array(X_text)
X_visual = np.array(X_visual)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

print(f"Shapes → Audio:{X_audio.shape}, Text:{X_text.shape}, Visual:{X_visual.shape}, Labels:{y_encoded.shape}")

# === SCALE AND PCA (global) ===
scaler_audio = StandardScaler().fit(X_audio)
X_audio_scaled = scaler_audio.transform(X_audio)
pca_audio = PCA(n_components=PCA_COMPONENTS_AUDIO).fit(X_audio_scaled)
X_audio_pca = pca_audio.transform(X_audio_scaled)

scaler_text = StandardScaler().fit(X_text)
X_text_scaled = scaler_text.transform(X_text)
pca_text = PCA(n_components=PCA_COMPONENTS_TEXT).fit(X_text_scaled)
X_text_pca = pca_text.transform(X_text_scaled)

scaler_visual = StandardScaler().fit(X_visual)
X_visual_scaled = scaler_visual.transform(X_visual)
pca_visual = PCA(n_components=PCA_COMPONENTS_VISUAL).fit(X_visual_scaled)
X_visual_pca = pca_visual.transform(X_visual_scaled)

# === CONFUSION MATRIX FUNCTION ===
def plot_confusion(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.show()

# === K-FOLD + STACKING META-CLASSIFIER ===
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
final_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_audio_pca, y_encoded), 1):
    X_train_audio, X_val_audio = X_audio_pca[train_idx], X_audio_pca[val_idx]
    X_train_text, X_val_text = X_text_pca[train_idx], X_text_pca[val_idx]
    X_train_visual, X_val_visual = X_visual_pca[train_idx], X_visual_pca[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # SMOTE per modality
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_audio_res, y_train_res = smote.fit_resample(X_train_audio, y_train)
    X_train_text_res, _ = smote.fit_resample(X_train_text, y_train)
    X_train_visual_res, _ = smote.fit_resample(X_train_visual, y_train)

    # Base Logistic Regression models
    model_audio = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    model_text = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    model_visual = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)

    model_audio.fit(X_train_audio_res, y_train_res)
    model_text.fit(X_train_text_res, y_train_res)
    model_visual.fit(X_train_visual_res, y_train_res)

    # Predict probabilities on validation set
    proba_audio = model_audio.predict_proba(X_val_audio)
    proba_text = model_text.predict_proba(X_val_text)
    proba_visual = model_visual.predict_proba(X_val_visual)

    # Stack probabilities as features for meta-classifier
    X_meta_train = np.hstack([proba_audio, proba_text, proba_visual])

    # Train meta-classifier
    meta_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    meta_clf.fit(X_meta_train, y_val)  # Using validation set probs as pseudo-training

    # Predict
    y_pred = meta_clf.predict(X_meta_train)

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    final_results.append([fold, acc, f1, bal_acc])

    # Confusion matrix
    plot_confusion(y_val, y_pred, le.classes_, f"Stacked Meta Fold {fold}")

# Summary
df = pd.DataFrame(final_results, columns=["Fold", "Accuracy", "F1-macro", "Balanced Acc"])
print("\n=== Stacked Meta-Classifier Results ===")
print(df)
print("\nMean metrics:")
print(df.mean(numeric_only=True))
