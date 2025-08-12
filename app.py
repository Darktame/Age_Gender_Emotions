import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from collections import deque

# Quiet down TensorFlow and oneDNN noise before importing TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Silence OpenCV logs if supported
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

# Load models
try:
    age_gender_model = load_model("models/age_gender.h5", compile=False)
    emotion_model    = load_model("models/emotion.h5", compile=False)
except Exception as e:
    print(f"Error loading Keras models: {e}")
    sys.exit(1)

# Labels
gender_labels  = ["Male", "Female"]
emotion_labels = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Face detector
try:
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
except Exception as e:
    print(f"Error loading face detector: {e}")
    sys.exit(1)

# Buffers
age_buffer = deque(maxlen=10)
max_expansion_pixels = 50

# Detect expected emotion model input shape (robust to channels-first/last)
emotion_input_shape = emotion_model.input_shape
try:
    shape = emotion_input_shape
    if isinstance(shape, list) and len(shape) == 1:
        shape = shape[0]
    if isinstance(shape, (list, tuple)) and len(shape) >= 4 and shape[-1] in (1, 3):
        emotion_channels = int(shape[-1])
    elif isinstance(shape, (list, tuple)) and len(shape) >= 4 and shape[1] in (1, 3):
        emotion_channels = int(shape[1])
    else:
        emotion_channels = 1
except Exception:
    emotion_channels = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive Base64 image
        if not request.is_json or 'image' not in request.json:
            return jsonify({"error": "Missing 'image' in JSON body"}), 400
        data = request.json['image']
        encoded = data.split(',')[1]  # remove data:image/jpeg;base64,
        try:
            img_bytes = base64.b64decode(encoded)
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

        # Convert to NumPy array
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        H, W  = frame.shape[:2]

        # Face detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        results = []

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.7:
                continue

            box = (detections[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
            x, y, x2, y2 = np.clip(box, [0, 0, 0, 0], [W, H, W, H])
            w, h = x2 - x, y2 - y
            if w < 20 or h < 20:
                continue

            tm = min(int(h * 1.0), max_expansion_pixels)
            bm = min(int(h * 0.5), max_expansion_pixels)
            sm = min(int(w * 0.3), max_expansion_pixels)
            x1 = max(0, x - sm)
            y1 = max(0, y - tm)
            x2 = min(W, x + w + sm)
            y2 = min(H, y + h + bm)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            fh, fw = face.shape[:2]
            if fh > fw:
                pad = (fh - fw) // 2
                face_padded = cv2.copyMakeBorder(face, 0, 0, pad, fh - fw - pad,
                                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif fw > fh:
                pad = (fw - fh) // 2
                face_padded = cv2.copyMakeBorder(face, pad, fw - fh - pad, 0, 0,
                                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                face_padded = face

            g = cv2.resize(face_padded, (128, 128)).astype("float32") / 255.0
            g = np.expand_dims(g, axis=0)
            age_p, gender_p = age_gender_model.predict(g, verbose=0)
            age = int(age_p[0][0])
            gender = gender_labels[int(gender_p[0][0] > 0.5)]
            age_buffer.append(age)
            avg_age = int(np.mean(age_buffer))

            if emotion_channels == 1:
                face_gray = cv2.cvtColor(face_padded, cv2.COLOR_BGR2GRAY)
                e = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
                e = np.expand_dims(e, axis=-1)
            else:
                face_rgb = cv2.cvtColor(face_padded, cv2.COLOR_BGR2RGB)
                e = cv2.resize(face_rgb, (48, 48)).astype("float32") / 255.0

            e = np.expand_dims(e, axis=0)
            emotion_idx = np.argmax(emotion_model.predict(e, verbose=0))
            emotion = emotion_labels[emotion_idx]

            results.append({
                "age": avg_age,
                "gender": gender,
                "emotion": emotion
            })

        return jsonify(results=results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
