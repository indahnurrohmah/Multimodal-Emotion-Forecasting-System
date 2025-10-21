import os
import pandas as pd
import requests
from tqdm import tqdm
import subprocess
import gdown

# === CONFIG ===
TRAIN_CSV = "broken_train.csv"
# TEST_CSV = "broken_test.csv"
TEST_CSV = "datatest_revisi.csv"
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
BASE_DIR = "dataset"

INSTAGRAM_USER = os.getenv("yourusername")      # set with: set IG_USER=yourusername
INSTAGRAM_PASS = os.getenv("yourpassword")      # set with: set IG_PASS=yourpassword

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)



def is_video_valid(filepath):
    """Check if video can be opened with ffprobe."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries",
               "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def download_file(url, save_path):
    """Download video depending on source (YouTube, Reels, Drive, or direct)."""
    try:
        # Case 1: YouTube or Instagram Reels → use yt-dlp
        if "youtube.com" in url or "youtu.be" in url or "instagram.com/reel" in url:
            cmd = [
                "yt-dlp",
                "--cookies", "cookies.txt",
                "-o", save_path,
                url
            ]
            # Instagram may need login
            if "instagram.com" in url:
                cmd.extend(["--username", INSTAGRAM_USER, "--password", INSTAGRAM_PASS])
            subprocess.run(cmd, check=True)
            return True

        # Case 2: Google Drive
        elif "drive.google.com" in url:
            if "id=" in url:
                file_id = url.split("id=")[-1]
            else:
                file_id = url.split("/d/")[1].split("/")[0]

            gdown.download(f"https://drive.google.com/uc?id={file_id}", save_path, quiet=False)
            return True

        # Case 3: Direct .mp4 or other link
        else:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True

    except Exception as e:
        print(f"❌ Failed: {url} ({e})")
        return False


def safe_download(url, save_path, max_retries=3):
    """Download and check validity. Retry if broken."""
    for attempt in range(max_retries):
        if download_file(url, save_path):
            if is_video_valid(save_path):
                return True
            else:
                print(f"⚠️ Invalid/corrupt video: {save_path}. Retrying...")
                try:
                    os.remove(save_path)  # remove broken file
                except FileNotFoundError:
                    pass
        else:
            print(f"⚠️ Download failed: {url}. Retrying...")
    return False


# === DOWNLOAD TRAIN VIDEOS ===
print("\n📥 Downloading train videos...")
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    url = str(row["video"])         # adjust column name if needed
    label = str(row["emotion"])
    video_id = str(row["id"])

    label_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    filename = os.path.join(label_dir, f"{video_id}.mp4")

    # ✅ Check file existence + validity
    if not os.path.exists(filename) or not is_video_valid(filename):
        if os.path.exists(filename):  # delete broken file
            print(f"⚠️ Removing broken file: {filename}")
            os.remove(filename)
        download_file(url, filename)


# === DOWNLOAD TEST VIDEOS ===
print("\n📥 Downloading test videos...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    url = str(row["video"])         # adjust column name if needed
    video_id = str(row["id"])

    filename = os.path.join(TEST_DIR, f"{video_id}.mp4")

    if not os.path.exists(filename) or not is_video_valid(filename):
        if os.path.exists(filename):  # delete broken file
            print(f"⚠️ Removing broken file: {filename}")
            os.remove(filename)
        download_file(url, filename)

print("\n✅ Download complete! All valid files saved in:", BASE_DIR)
