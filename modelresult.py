import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# === CONFIG ===
TRAIN_DIR = "./features"   # your training JSON folder
TEST_DIR = "test_features"         # your test JSON folder
PCA_COMPONENTS_AUDIO = 128
PCA_COMPONENTS_TEXT = 128
PCA_COMPONENTS_VISUAL = 64

# Mapping for emotions to numbers
emotion_to_num = {
    "Proud":0, "Trust":1, "Joy":2, "Surprise":3,
    "Neutral":4, "Sadness":5, "Fear":6, "Anger":7
}

# === LOAD TRAIN DATA ===
X_audio, X_text, X_visual, y = [], [], [], []

for fname in os.listdir(TRAIN_DIR):
    if not fname.endswith(".json"): continue
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
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
    y.append(emotion_to_num.get(sample.get("emotion"), 4))  # default to Neutral if missing

X_audio = np.array(X_audio)
X_text = np.array(X_text)
X_visual = np.array(X_visual)
y = np.array(y)

# === SCALE AND PCA ===
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

# === APPLY SMOTE ON TRAIN DATA ===
smote = SMOTE(random_state=42, k_neighbors=1)
X_audio_res, y_res = smote.fit_resample(X_audio_pca, y)
X_text_res, _ = smote.fit_resample(X_text_pca, y)
X_visual_res, _ = smote.fit_resample(X_visual_pca, y)

# === TRAIN BASE MODELS ===
model_audio = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
model_text = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
model_visual = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)

model_audio.fit(X_audio_res, y_res)
model_text.fit(X_text_res, y_res)
model_visual.fit(X_visual_res, y_res)

# === CREATE META-CLASSIFIER TRAIN DATA ===
proba_audio = model_audio.predict_proba(X_audio_pca)
proba_text = model_text.predict_proba(X_text_pca)
proba_visual = model_visual.predict_proba(X_visual_pca)

X_meta_train = np.hstack([proba_audio, proba_text, proba_visual])
meta_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
meta_clf.fit(X_meta_train, y)  # final meta-classifier

# === PREDICT ON TEST DATA ===
test_ids, preds = [], []

for fname in os.listdir(TEST_DIR):
    if not fname.endswith(".json"): continue
    test_ids.append(fname.replace(".json",""))
    
    with open(os.path.join(TEST_DIR, fname), "r", encoding="utf-8") as f:
        sample = json.load(f)
    
    audio_vec = np.array(sample.get("audio", []), dtype=float)
    text_vec = np.array(sample.get("text", {}).get("embedding", []), dtype=float)
    visual_vec = np.array(sample.get("visual", []), dtype=float)
    
    if audio_vec.size == 0: audio_vec = np.zeros(768)
    if text_vec.size == 0: text_vec = np.zeros(768)
    if visual_vec.size == 0: visual_vec = np.zeros(512)
    
    # SCALE + PCA
    audio_p = pca_audio.transform(scaler_audio.transform(audio_vec.reshape(1,-1)))
    text_p = pca_text.transform(scaler_text.transform(text_vec.reshape(1,-1)))
    visual_p = pca_visual.transform(scaler_visual.transform(visual_vec.reshape(1,-1)))
    
    # BASE PROBABILITIES
    proba_a = model_audio.predict_proba(audio_p)
    proba_t = model_text.predict_proba(text_p)
    proba_v = model_visual.predict_proba(visual_p)
    
    # STACK AND META-PREDICT
    X_meta_test = np.hstack([proba_a, proba_t, proba_v])
    pred = meta_clf.predict(X_meta_test)[0]
    preds.append(pred)

# === CREATE DATAFRAME AND SORT BY ID NUMERICALLY ===
submission = pd.DataFrame({"id": test_ids, "predicted": preds})
submission["id"] = submission["id"].astype(int)  # convert to int
submission = submission.sort_values("id").reset_index(drop=True)

# === SAVE CSV ===
submission.to_csv("submission+BDCUGM03.csv", index=False)
print("✅ Submission saved as submission+BDCUGM03.csv (sorted by id)")
