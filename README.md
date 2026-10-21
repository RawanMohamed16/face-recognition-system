# Face Recognition System

A Python face recognition system that identifies people in photos using face embeddings, evaluated on a held-out test set and wrapped in a Streamlit web app for interactive use.

Built with `face_recognition` (dlib), OpenCV, scikit-learn, and Streamlit. Developed and run in Google Colab.

## Features

- Face detection and 128-dimensional face encoding with `face_recognition` (dlib)
- Identity matching by nearest-neighbour Euclidean distance, with a tolerance threshold (0.6) to label unmatched faces as **Unknown**
- Stratified 80/20 train/test split by person, so every identity is represented in both sets
- Evaluation on held-out images with accuracy and a per-person classification report
- Streamlit web app with image upload, webcam snapshot, and model information modes

## How It Works

1. **Split:** `Dataset.csv` (image id and person label) is split 80/20, stratified by label. People with only one image go entirely into the training set, since a single image cannot be both trained on and tested.
2. **Encode:** each training image is passed through the face detector, and the first detected face is converted into a 128-dimensional encoding. Encodings and labels are saved to `encodings.pkl`.
3. **Recognize:** for a new image, each detected face is encoded and compared with all stored encodings. The closest match is returned if its distance is at most the tolerance (0.6); otherwise the face is labelled `Unknown`. Confidence is shown as `(1 - distance) * 100`.
4. **Evaluate:** the same procedure runs on the held-out test images, and predictions are compared with the true labels.

## Dataset

- Source: **[add dataset name and link here]**
- Size: **[total images]** images of **[number]** people
- Split: 80% training (2,049 training images) / 20% test, random seed 42

The images are **not included** in this repository. Download the dataset from the source above and follow the setup steps below.

## Evaluation

The notebook includes a batch evaluation cell that runs the recognizer on the held-out test images and reports overall accuracy and a per-person classification report (precision, recall, F1). Run it after encoding the training set to reproduce the results.

## Web App

The Streamlit app (`app.py`, written by the notebook) has three modes:

- **Image Upload & Recognition:** upload a photo and see the detected faces with names drawn on the image
- **Webcam Snapshot Mode:** capture a snapshot and run recognition on it
- **Model Information:** details about the loaded encodings

In Colab, the app is exposed through an ngrok tunnel. You need your own free ngrok account and auth token. Store the token as a Colab secret and do not paste it into the notebook.

## Setup

1. Open `face_recognition_project.ipynb` in Google Colab.
2. In your Google Drive, create a folder named `face_recognition_project` containing:
   - the dataset zip, named `Original Images.zip`
   - `Dataset.csv` with the columns `id` and `label`
3. Run the notebook cells in order. The install cell takes a few minutes because `dlib` is compiled.
4. Encoding the training images is the slowest step. The result is saved to Drive as `encodings.pkl`, so it only has to run once.

Python dependencies are listed in `requirements.txt`:

```
face-recognition
opencv-python
dlib
pandas
scikit-learn
tqdm
matplotlib
seaborn
streamlit
pyngrok
```

## Project Structure

```
face-recognition-system/
├── face_recognition_project.ipynb   # split, encoding, inference, evaluation, app launch
├── requirements.txt
└── README.md
```

Generated at run time and not committed: `encodings.pkl`, `train_labels.csv`, `test_labels.csv`, `app.py`.

## Limitations

- Matching uses a fixed tolerance of 0.6. A lower value reduces false matches but rejects more genuine faces; a higher value does the opposite.
- Only the first detected face in each training image is encoded.
- Accuracy depends on lighting, pose, image quality, and the number of images per person. People with very few images are harder to match.
- The system is designed for a closed set of known people. Performance on people outside the dataset has not been evaluated beyond the `Unknown` threshold.
- Face recognition raises privacy and bias concerns. This project is for learning and demonstration, not for identifying real people without their consent.

## Tech Stack

Python, face_recognition (dlib), OpenCV, NumPy, pandas, scikit-learn, matplotlib, seaborn, Streamlit, ngrok

## Author

Rawan Abdelsalam Mohamed: Computer Science student at AASTMT

