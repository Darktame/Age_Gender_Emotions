import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load pre-trained models
age_gender_model = load_model("age_gender_best_model.h5", compile=False)
emotion_model    = load_model("last_cnn_1.h5", compile=False)

# Labels
gender_labels  = ["Male", "Female"]
emotion_labels = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Video setup
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Buffers
age_buffer = deque(maxlen=10)
max_expansion_pixels = 50

# Detect expected emotion model input shape
emotion_input_shape = emotion_model.input_shape  
emotion_channels = 1 if emotion_input_shape[-1] == 1 else 3

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
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
            e = np.expand_dims(e, axis=-1)  # (48,48,1)
        else:
            
            face_rgb = cv2.cvtColor(face_padded, cv2.COLOR_BGR2RGB)
            e = cv2.resize(face_rgb, (48, 48)).astype("float32") / 255.0

        e = np.expand_dims(e, axis=0)  # Batch dimension
        emotion_idx = np.argmax(emotion_model.predict(e, verbose=0))
        emotion = emotion_labels[emotion_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{avg_age}, {gender}, {emotion}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        ty = y1 - 8 if y1 - th - 10 >= 0 else y2 + th + 8
        by = y1 - th - 10 if y1 - th - 10 >= 0 else y2 + 10

        cv2.rectangle(frame, (x1, by), (x1 + tw, ty + 2), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Age · Gender · Emotion Detection", frame)
    if cv2.waitKey(25) & 0xFF == ord('x'):
        break

video.release()
cv2.destroyAllWindows()
