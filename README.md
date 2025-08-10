#  Age, Gender & Emotions Detection 

A deep learning project to analyze human faces and predict:
- **Age group** (0-80)
- **Gender** (male / female)
- **Emotion**(['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])

Built with TensorFlow/Keras and OpenCV, this project offers pre-trained models ready for inference.


##  Features
- Detects **age**, **gender**, and **emotion** from a single image or webcam feed
- Pre-trained `.h5` models included for quick use
- Face detection using OpenCV (Haar Cascades or DNN)
- Real-time or image-based processing

## Datasets Used
UTKFace – age & gender
FER2013 – emotion detection

## Age-Gender-Emotion-Detection
── .gitattributes                 # Git LFS tracking config for large files
── Emotion_detection_best_model.h5 # Trained model for emotion detection
── age_gender_best_model.h5        # Trained model for age & gender detection
── age-gender-detection.ipynb      # Notebook for age & gender detection
── emotion-detection.ipynb         # Notebook for emotion detection
── cnn_project.py                  # Python script combining all detections (live detection)
── deploy.prototxt                 # Face detection architecture file
── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained face detection weights
── Requirements.txt                # Required Python dependencies
── README.md                       # Project documentation
