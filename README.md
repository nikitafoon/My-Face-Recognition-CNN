# Real-Time Face Recognition via Metric Learning 📸🧠

An end-to-end, custom-built face recognition system powered by **PyTorch**, **PyTorch Lightning**, and **OpenCV**. This project demonstrates the implementation of a Convolutional Neural Network (CNN) from scratch to solve a Metric Learning problem using **Triplet Margin Loss**, allowing the system to verify a specific user's identity in real-time via a webcam.

## 🚀 Project Overview

Unlike traditional classification networks that predict a fixed set of classes, this project uses **Metric Learning** (Open-set Face Recognition). The network is trained to map face images into a 128-dimensional Euclidean space (embeddings) where distances directly correspond to face similarity. 

### Key Features:
*   **Custom CNN Architecture:** Built from scratch without using pre-trained backbones (like ResNet or MobileNet).
*   **Offline Triplet Mining:** Custom PyTorch `Dataset` that dynamically generates (Anchor, Positive, Negative) triplets for training.
*   **PyTorch Lightning Integration:** Clean, scalable training loop with automatic checkpointing and Early Stopping.
*   **Real-time Inference:** OpenCV integration for face detection (Haar Cascades) and PyTorch for identity verification on a live webcam feed.
*   **Automated Face Cropping:** Uses Google's MediaPipe for precise, automated bounding-box extraction from raw target photos to construct a high-quality Anchor dataset.

---

## 🏗️ Architecture & Methodology

### 1. Data Pipeline & Triplet Generation
The dataset consists of two main parts:
*   **Target (Anchor & Positive):** A custom dataset of my own face (`my_face_dataset/`).
*   **Control (Negative):** Random identities from the `img_align_celeba` dataset.

Instead of hardcoding CSV labels, the custom `MyDataset` class dynamically samples images. For every training step, it fetches:
1.  **Anchor:** A random photo of the target.
2.  **Positive:** Another random photo of the target.
3.  **Negative:** A photo of a random person from the control dataset.

Data augmentation (`Albumentations`) is heavily applied to the training set (HorizontalFlips, ShiftScaleRotate, Normalize) to prevent overfitting and simulate webcam domain noise.

#### Automated Target Preprocessing (MediaPipe Integration)
To ensure the target dataset (Anchor/Positive) is perfectly aligned and isolated from the background, a dedicated Jupyter Notebook script (`datasetPrepare.ipynb`) was implemented[cite: 2]. This script automatically processes a raw folder of webcam photos (`/home/nikita/Pictures/Camera`) and outputs tightly cropped faces[cite: 2]:

*   **Detector:** Utilizes `mediapipe.tasks.vision.FaceDetector` loaded with the `blaze_face_full_range.tflite` model for high-accuracy face localization[cite: 2].
*   **Cropping Logic:** The `visualize` function extracts the bounding box coordinates (`origin_x`, `origin_y`, `width`, `height`) from the MediaPipe detection results and slices the NumPy array of the original image[cite: 2].
*   **Batch Processing:** It iterates through the entire source directory, converts the cropped arrays from BGR to RGB, and saves the prepared `.jpg` files directly into the training folder (`my_face_dataset/`) with a `cropped_` prefix[cite: 2].

### 2. The Neural Network (`MyCNN`)
A lightweight 4-block Convolutional Neural Network designed to process `128x128` RGB images and output a normalized 128-dimensional embedding.

*   **Feature Extractor:** 4 sequential blocks of `Conv2d` (3x3 kernels) -> `ReLU` -> `MaxPool2d`. Channels expand as 16 -> 32 -> 64 -> 64.
*   **Head:** `Flatten` -> `Linear` -> `ReLU` -> `BatchNorm1d` -> `Dropout(0.2)` -> `Linear` -> `ReLU` -> `BatchNorm1d` -> `Dropuot(0.1)` -> `Linear`.
*   **L2 Normalization:** Applied in the forward pass of the Lightning Module to project embeddings onto a hypersphere, stabilizing the Triplet Loss.

### 3. Loss Function: Triplet Margin Loss
The network optimizes the `nn.TripletMarginLoss(margin=1.0, p=2)`.
It ensures that the distance between the Anchor and Positive (D_pos) is smaller than the distance between the Anchor and Negative (D_neg) by at least the specified `margin`.

---

## 🛠️ Installation & Setup

**Requirements:**
*   Python 3.10+
*   PyTorch & PyTorch Lightning
*   OpenCV (`opencv-python`)
*   Albumentations
*   Scikit-learn, Torchmetrics
*   MediaPipe (`mediapipe`) - Required for the data preprocessing notebook.
```bash
# Clone the repository
git clone [https://github.com/yourusername/custom-face-recognition.git](https://github.com/yourusername/custom-face-recognition.git)
cd custom-face-recognition

# Install dependencies
pip install torch torchvision torchaudio pytorch-lightning opencv-python albumentations torchmetrics scikit-learn mediapipe matplotlib
```

**Directory Structure:**
Ensure your data is organized as follows before training:
```text
project_root/
├── my_face_dataset/       # Your cropped personal photos (.jpg)
├── img_align_celeba/      # CelebA dataset photos (.jpg)
├── datasetPrepare.ipynb   # MediaPipe script to extract faces from raw photos
└── faceRecognitionCNN.py  # Real-time webcam script
```

---

## 💻 Usage

### 1. Data Preprocessing
Before training, place your raw photos in a designated folder and run `datasetPrepare.ipynb` to automatically detect, crop, and move your faces into `my_face_dataset/`. Ensure the `blaze_face_full_range.tflite` model is downloaded and the paths in the notebook match your system configuration.

### 2. Training the Model & Real-Time Webcam Inference
*Note: I've achived greate results with ~800 selfies in my dataset.*  

`faceRecognitionCNN.ipynb` contain all what you need for these tasks. 

To train the model from scratch, run the script. The script handles `train_test_split` (preventing data leakage by splitting CelebA by identity folders) and launches the PyTorch Lightning Trainer.

Once trained, the best model weights are saved in the `my_models/` directory. Run the inference script to test the model on your webcam. *Note: You must specify a `REFERENCE_PHOTO_PATH` (a clear image of your face) inside the script to serve as the Anchor embedding.*  
**Controls:** Press `q` to quit the webcam window.

---

## 🧠 Engineering Challenges & Solutions

During development, several deep learning challenges were addressed:

1.  **Model Collapse (Vanishing Gradients):** 
    Initially, the network predicted the exact same embedding for every image. This was resolved by drastically lowering the Learning Rate (to `1e-4`) and temporarily disabling `Dropout` during the warm-up phase.
2.  **Hardcoded Accuracy Metric vs. Triplet Loss:** 
    Standard `Accuracy` (binary) fails in Metric Learning without a threshold. I implemented a dynamic distance logger in TensorBoard to physically observe the vector separation. By tracking positive and negative distances directly, I could manually calculate the optimal decision boundary.
3.  **Domain Shift (Sim2Real):** 
    The model achieved 99.7% accuracy on the test, but failed on real-world recognition, so I decided to slightly change inference threshold from 0.5 to 0.4. It was enough to make my model recognize me and not recognize my wife :)

---
*Author: Nikita Mefodovskiy*
```
