#  Age, Gender & Emotions Detection 

A deep learning project to analyze human faces and predict:
- **Age group** - 0 to 80
- **Gender** - Male, Female
- **Emotion** - Angry, Fearful, Happy, Neutral, Sad, Surprised

Built with TensorFlow and OpenCV, this project offers pre-trained models ready for inference.

## Usage
Install dependencies
```sh
$ pip install -r requirements.txt
```

### Webapp
```sh
$ flask run
```

### Application
```sh
$ python cnn_project.py
```

##  Features
- Detects **age**, **gender**, and **emotion** from a single image or webcam feed
- Pre-trained `.h5` models included for quick use
- Face detection using OpenCV (Haar Cascades or DNN)
- Real-time or image-based processing

## Datasets Used
- UTKFace – Age and gender
- FER2013 – Emotion detection
