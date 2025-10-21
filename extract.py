import os
import json
import librosa
import torch
import numpy as np
import cv2
import csv
import whisper
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoTokenizer, AutoModel
import mediapipe as mp
import subprocess
from torchvision.models import resnet18, ResNet18_Weights
import imageio_ffmpeg as ffmpeg
import easyocr
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg.get_ffmpeg_exe())
# ==============================
# CONFIG
# ==============================
VIDEO_ROOT = "dataset/train/"   # parent directory containing emotion folders
# VIDEO_ROOT = "dataset/test/"   # parent directory containing emotion folders
OUTPUT_DIR = "features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def needs_audio_reextract(features: dict) -> bool:
    """
    Return True if audio field uses old schema (mfcc + wav2vec2 dict).
    """
    audio = features.get("audio")
    if isinstance(audio, dict):
        if "mfcc" in audio and "wav2vec2" in audio:
            return True
    return False

# ==============================
# TEXT-VIDEO FEATURE EXTRACTION
# ==============================
def extract_text_from_frames(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    texts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray)

            for (bbox, text, conf) in results:
                if conf > 0.5 and text.strip():
                    texts.append(text.strip())
        frame_count += 1

    cap.release()

    # Merge all detected texts
    combined_text = " ".join(texts).strip()

    if combined_text:
        tokens = text_tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = text_model(**tokens)
            text_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    else:
        text_embedding = np.array([])

    return {
        "transcript": combined_text,
        "embedding": text_embedding.tolist()
    }

# ==============================
# AUDIO FEATURE EXTRACTION
# ==============================
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

def extract_audio_from_video(video_path, sr=16000):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-ar", str(sr), "-ac", "1", audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return audio_path
    except Exception as e:
        print(f"⚠️ Failed to extract audio with ffmpeg: {e}")
        return None

# def extract_audio_features(video_path):
#     audio_path = extract_audio_from_video(video_path)
#     if not audio_path:
#         return {"mfcc": [], "wav2vec2": []}

#     try:
#         y, sr = librosa.load(audio_path, sr=16000)
#     except Exception as e:
#         print(f"⚠️ Could not load audio for {video_path}: {e}")
#         return {"mfcc": [], "wav2vec2": []}

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfcc_mean = mfcc.mean(axis=1)

#     inputs = wav2vec_processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         embeddings = wav2vec_model(inputs.input_values.to(device)).last_hidden_state.mean(dim=1).cpu().numpy().flatten()

#     return {
#         "mfcc": mfcc_mean.tolist(),
#         "wav2vec2": embeddings.tolist()
#     }

def extract_audio_features(path, chunk_size=10, sr=16000, min_duration=0.3):  
    # 🔹 Extract WAV from video
    wav_path = extract_audio_from_video(path, sr)

    if wav_path is None or not os.path.exists(wav_path):
        print(f"❌ No audio extracted for {path}")
        return None

    # 🔹 Load audio at 16kHz
    audio, sr = librosa.load(wav_path, sr=sr)

    # 🔹 Skip if audio is too short
    duration = len(audio) / sr
    if duration < min_duration:
        print(f"⚠️ Skipping {path}, audio too short ({duration:.2f}s)")
        return None

    samples_per_chunk = chunk_size * sr
    embeddings = []

    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i:i+samples_per_chunk]
        if len(chunk) < int(0.1 * sr):  # <0.1s chunk, skip
            continue

        # Convert chunk to tensor
        inputs = wav2vec_processor(
            chunk, sampling_rate=sr, return_tensors="pt", padding=True
        )

        try:
            with torch.no_grad():
                outputs = wav2vec_model(inputs.input_values.to(device))
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                embeddings.append(emb)
        except RuntimeError as e:
            print(f"⚠️ Skipping problematic chunk in {path}: {e}")
            continue

    if len(embeddings) == 0:
        print(f"⚠️ No embeddings extracted for {path}")
        return None

    return np.mean(embeddings, axis=0)


# ==============================
# TEXT FEATURE EXTRACTION (Whisper + XLM-R)
# ==============================
whisper_model = whisper.load_model("base", device=device)

text_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
text_model = AutoModel.from_pretrained("xlm-roberta-base").to(device)

def extract_text_features(video_path):
    audio_path = extract_audio_from_video(video_path)
    if not audio_path:
        return {"transcript": "", "embedding": []}

    try:
        result = whisper_model.transcribe(audio_path, language=None)  # use WAV, not MP4
        transcript = result["text"].strip()
    except Exception as e:
        print(f"⚠️ Whisper failed for {video_path}: {e}")
        return {"transcript": "", "embedding": []}

    if transcript:
        tokens = text_tokenizer(transcript, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = text_model(**tokens)
            text_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    else:
        text_embedding = np.array([])

    return {
        "transcript": transcript,
        "embedding": text_embedding.tolist()
    }


# ==============================
# VISUAL FEATURE EXTRACTION
# ==============================
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

def extract_visual_features(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    embeddings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
                    x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)
                    face = frame[max(0,y1):y2, max(0,x1):x2]
                    if face.size > 0:
                        face_resized = cv2.resize(face, (224, 224))
                        face_tensor = torch.tensor(face_resized).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
                        with torch.no_grad():
                            emb = resnet(face_tensor).cpu().numpy().flatten()
                        embeddings.append(emb)
        frame_count += 1

    cap.release()
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()
    else:
        return []
    
import csv

def generate_incomplete_report(features_dir=OUTPUT_DIR, csv_path="incomplete_report.csv"):
    rows = []
    for fname in os.listdir(features_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(features_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                features = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load {fname}: {e}")
            rows.append({
                "id": fname.replace(".json", ""),
                "video": "UNKNOWN",
                "emotion": "UNKNOWN",
                "missing": "corrupt_json"
            })
            continue

        missing = is_incomplete(features)
        if missing:  # only log incomplete ones
            rows.append({
                "id": features.get("id", fname.replace(".json", "")),
                "video": features.get("video", "UNKNOWN"),
                "emotion": features.get("emotion", "UNKNOWN"),
                "missing": ",".join(missing)
            })

    # Write CSV
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "video", "emotion", "missing"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Incomplete report saved to {csv_path} ({len(rows)} entries)")
    else:
        print("🎉 All JSON files are complete! No CSV generated.")


# ==============================
# MAIN PIPELINE
# ==============================
# 

# Custom converter for numpy arrays + torch tensors
def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().tolist()  # convert tensor -> list
    return val

# def is_incomplete(features: dict) -> list:
#     """Return list of missing/empty feature types."""
#     missing = []
#     for key in ["audio", "text", "visual"]:
#         if key not in features or features[key] is None or features[key] == [] or features[key] == {}:
#             missing.append(key)
#     return missing
def is_incomplete(features: dict) -> list:
    """Return list of missing/empty feature types. 
    If text exists but transcript/embedding empty → treat as missing."""
    missing = []
    
    # Check audio
    if not features.get("audio"):  # [] or None
        missing.append("audio")
    
    # Check visual
    if not features.get("visual"):  # [] or None
        missing.append("visual")
    
    # Check text (dict case)
    text = features.get("text")
    if not text or (
        not text.get("transcript") and not text.get("embedding")
    ):
        # transcript is "" or None AND embedding is [] or None
        missing.append("text")
    
    return missing



# # TEST
# import os
# import json
# from tqdm import tqdm

# def process_videos(root_dir):
#     for fname in tqdm(os.listdir(root_dir), desc="Processing videos"):
#         if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
#             continue

#         video_path = os.path.join(root_dir, fname)
#         video_path = os.path.abspath(video_path).replace("\\", "/")
#         video_id = os.path.splitext(fname)[0]
#         out_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")

#         features = None
#         missing = None

#         # case 1: no JSON yet → extract everything
#         if not os.path.exists(out_path):
#             print(f"▶️ New file {video_id}, extracting all features...")
#             features = {
#                 "id": video_id,
#                 "video": fname,
#                 "audio": extract_audio_features(video_path),
#                 "text": extract_text_features(video_path),
#                 "visual": extract_visual_features(video_path)
#             }

#         else:
#             # case 2: JSON exists → try loading it
#             try:
#                 with open(out_path, "r", encoding="utf-8") as f:
#                     features = json.load(f)

#                 missing = is_incomplete(features)
#             except json.JSONDecodeError:
#                 print(f"⚠️ Corrupt JSON for {video_id}, re-extracting everything...")
#                 features = {
#                     "id": video_id,
#                     "video": fname,
#                     "audio": extract_audio_features(video_path),
#                     "text": extract_text_features(video_path),
#                     "visual": extract_visual_features(video_path)
#                 }
#                 missing = None

#             # if JSON is valid but incomplete → fix only missing parts
#             if missing:
#                 print(f"♻️ Re-extracting {video_id}, missing: {missing}")
#                 if "audio" in missing:
#                     features["audio"] = extract_audio_features(video_path)
#                 if "text" in missing:
#                     features["text"] = extract_text_features(video_path)
#                 if "visual" in missing:
#                     features["visual"] = extract_visual_features(video_path)

#             elif missing == []:
#                 print(f"⏩ Skipping {video_id}, already complete.")
#                 continue

#         # save updated/complete JSON
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(features, f, indent=2, ensure_ascii=False, default=to_serializable)


# TRAIN
def process_videos(root_dir):
    for emotion in os.listdir(root_dir):
        emotion_dir = os.path.join(root_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        for fname in tqdm(os.listdir(emotion_dir), desc=f"Processing {emotion}"):
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(emotion_dir, fname)
            video_path = os.path.abspath(video_path).replace("\\", "/")
            video_id = os.path.splitext(fname)[0]
            out_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")

            features = None
            missing = None

            # case 1: no JSON yet → extract everything
            if not os.path.exists(out_path):
                print(f"▶️ New file {video_id}, extracting all features...")
                features = {
                    "id": video_id,
                    "video": fname,
                    "emotion": emotion,
                    "audio": extract_audio_features(video_path),
                    "text": extract_text_features(video_path),
                    "visual": extract_visual_features(video_path)
                }

            else:
                # case 2: JSON exists → try loading it
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        features = json.load(f)

                    missing = is_incomplete(features)
                    if needs_audio_reextract(features):
                        print(f"🔄 {video_id} has old audio format, re-extracting audio...")
                        features["audio"] = extract_audio_features(video_path)
                            # 🔹 Save immediately so it doesn’t get skipped later
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(features, f, indent=2, ensure_ascii=False, default=to_serializable)

                except json.JSONDecodeError:
                    print(f"⚠️ Corrupt JSON for {video_id}, re-extracting everything...")
                    features = {
                        "id": video_id,
                        "video": fname,
                        "emotion": emotion,
                        "audio": extract_audio_features(video_path),
                        "text": extract_text_features(video_path),
                        "visual": extract_visual_features(video_path)
                    }
                    missing = None

                # if JSON is valid but incomplete → fix only missing parts
                if missing:
                    print(f"♻️ Re-extracting {video_id}, missing: {missing}")
                    if "audio" in missing:
                        features["audio"] = extract_audio_features(video_path)
                    if "text" in missing:
                        features["text"] = extract_text_from_frames(video_path)
                    if "visual" in missing:
                        features["visual"] = extract_visual_features(video_path)
                    
                elif missing == []:
                    print(f"⏩ Skipping {video_id}, already complete.")
                    continue

            # save updated/complete JSON
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(features, f, indent=2, ensure_ascii=False, default=to_serializable)

# def process_videos(root_dir):
#     for emotion in os.listdir(root_dir):
#         emotion_dir = os.path.join(root_dir, emotion)
#         if not os.path.isdir(emotion_dir):
#             continue

#         for fname in tqdm(os.listdir(emotion_dir), desc=f"Processing {emotion}"):
#             if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
#                 continue

#             video_path = os.path.join(emotion_dir, fname)
#             video_path = os.path.abspath(video_path).replace("\\", "/")
#             video_id = os.path.splitext(fname)[0]  
#             out_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")

#             # 🚨 File checker: skip if already processed
#             if os.path.exists(out_path):
#                 print(f"⏩ Skipping {video_id}, already extracted.")
#                 continue

#             print(f"▶️ Extracting features for {video_id} ({emotion})")

#             features = {
#                 "id": video_id,
#                 "video": fname,
#                 "emotion": emotion,
#                 "audio": extract_audio_features(video_path),
#                 "text": extract_text_features(video_path),
#                 "visual": extract_visual_features(video_path)
#             }

#             with open(out_path, "w", encoding="utf-8") as f:
#                 # json.dump(features, f, indent=2, ensure_ascii=False)
#                 json.dump(features, f, indent=2, ensure_ascii=False, default=to_serializable)


process_videos(VIDEO_ROOT)
generate_incomplete_report()

