"""
Downloads the required model files into /models.
Run this once before using main.py.
"""

import urllib.request
import os
import bz2

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

SHAPE_PREDICTOR_URL = (
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
)
SHAPE_PREDICTOR_DEST = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # OpenCV DNN face detector
    for filename, url in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")

    # dlib 68-point shape predictor (ships as .bz2, needs decompression)
    if os.path.exists(SHAPE_PREDICTOR_DEST):
        print(f"Already exists: {SHAPE_PREDICTOR_DEST}")
    else:
        bz2_path = SHAPE_PREDICTOR_DEST + ".bz2"
        print("Downloading shape_predictor_68_face_landmarks.dat.bz2 (~100MB)...")
        urllib.request.urlretrieve(SHAPE_PREDICTOR_URL, bz2_path)
        print("Decompressing...")
        with bz2.open(bz2_path, "rb") as f_in, open(SHAPE_PREDICTOR_DEST, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(bz2_path)
        print(f"Saved to {SHAPE_PREDICTOR_DEST}")

    print("\nAll models ready.")


if __name__ == "__main__":
    download_models()
