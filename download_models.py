"""
Downloads the required OpenCV DNN face detection model files into /models.
Run this once before using main.py.
"""

import urllib.request
import os

MODEL_DIR = "models"
FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
    print("\nAll models ready.")

if __name__ == "__main__":
    download_models()
