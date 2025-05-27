# BrainTumourDetection_usingCNN
A Convolutional Neural Network (CNN)-based classifier to detect brain tumors from MRI images using TensorFlow and Keras.

---

## 📦 Dataset

* **Source**: [Kaggle - navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* **Categories**:

  * `yes`: MRI scans with tumors
  * `no`: MRI scans without tumors

---

## 🧰 Tools & Libraries

* Python, NumPy, OpenCV
* Matplotlib, Seaborn
* Scikit-learn
* TensorFlow / Keras
* KaggleHub (for automated dataset download)

---

## ⚙️ Pipeline Overview

1. **Data Loading & Preprocessing**

   * Images converted to grayscale and resized to 128×128
   * Labels (`yes`=1, `no`=0) are one-hot encoded
   * Pixel values normalized to \[0, 1]

2. **Model Architecture**

   ```
   Input (128x128x1)
     ↓
   Conv2D (32 filters) + MaxPooling
     ↓
   Conv2D (64 filters) + MaxPooling
     ↓
   Conv2D (128 filters) + MaxPooling
     ↓
   Flatten → Dropout(0.5) → Dense(128)
     ↓
   Output (Dense, softmax)
   ```

3. **Training**

   * Optimizer: Adam
   * Loss: Categorical Crossentropy
   * Epochs: 10
   * Validation split: 20%

4. **Evaluation**

   * Accuracy curve
   * Confusion matrix
   * Classification report

---

## 📈 Output Samples

* **Training vs Validation Accuracy**
* **Confusion Matrix Visualization**

---

## 🚀 How to Run

1. Install required packages:

   ```bash
   pip install opencv-python matplotlib seaborn scikit-learn tensorflow kagglehub
   ```

2. Run the script:

   ```bash
   python brain_tumor_cnn.py
   ```

3. Make sure your Kaggle API is set up for `kagglehub` to fetch the dataset.

---

## 💡 To Improve

* Add data augmentation
* Try transfer learning with pre-trained models
* Increase training epochs for better performance
* Deploy using Flask/Streamlit for real-time predictions

---

## 📚 Credits

* Dataset: [Navoneel Chakrabarty on Kaggle](https://www.kaggle.com/navoneel)

---
