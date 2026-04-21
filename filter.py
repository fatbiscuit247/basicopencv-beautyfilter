import cv2
import numpy as np

# Load OpenCV's DNN face detector (ships with opencv-python)
_face_net = None
_landmark_net = None

def _get_face_net():
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )
    return _face_net


def detect_faces(frame):
    """Returns list of (x, y, w, h) bounding boxes for detected faces."""
    net = _get_face_net()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def apply_bilateral_smoothing(image, diameter=9, sigma_color=75, sigma_space=75):
    """Bilateral filter: smooths while preserving edges."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def build_face_mask(frame, face):
    """Creates an elliptical mask for the face region to localize smoothing."""
    x, y, w, h = face
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = (x + w // 2, y + h // 2)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # Feather the mask edges so blending looks natural
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask


def correct_skin_tone(image):
    """Subtle warmth boost + redness reduction in skin tones."""
    result = image.astype(np.float32)
    # Slight warmth: boost red/green, reduce blue very mildly
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.04, 0, 255)  # Red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.02, 0, 255)  # Green
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.97, 0, 255)  # Blue
    return result.astype(np.uint8)


def apply_beauty_filter(frame):
    """
    Main beauty filter pipeline:
    1. Detect faces using DNN model
    2. Build a feathered elliptical face mask
    3. Apply bilateral smoothing to face region only
    4. Apply subtle skin tone correction
    5. Composite smoothed face back onto original frame
    """
    output = frame.copy()
    faces = detect_faces(frame)

    if not faces:
        return output  # No face detected — return original frame unchanged

    for face in faces:
        # Step 1: Build feathered mask for this face
        mask = build_face_mask(frame, face)
        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

        # Step 2: Smooth the full frame then isolate with mask
        smoothed = apply_bilateral_smoothing(frame)
        smoothed = correct_skin_tone(smoothed)

        # Step 3: Alpha-composite: blend smoothed face onto original
        output = (frame.astype(np.float32) * (1 - mask_3ch) +
                  smoothed.astype(np.float32) * mask_3ch)
        output = output.astype(np.uint8)

    return output