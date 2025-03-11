# Hand Gesture Recognition using CNN

This project trains a **Convolutional Neural Network (CNN)** to classify hand gestures into 11 categories:

✅ Like  
✅ Dislike  
✅ Fist  
✅ One  
✅ Two Up  
✅ Three  
✅ Four  
✅ Rock  
✅ Peace  
✅ Stop  
✅ Call  

## 📌 Features
- Custom **CNN model** built from scratch.
- Implements **data augmentation** to improve generalization.
- Uses **categorical crossentropy loss** for multi-class classification.
- **Early stopping** to prevent overfitting.
- Saves the trained model as `hand_gesture_cnn.h5`.

## 📂 Dataset Structure
Organize images into subfolders under `Data/`, where each subfolder represents a gesture class:
```
Data/
│── Like/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── ...
│── Dislike/
│── Fist/
│── One/
│── Two Up/
│── Three/
│── Four/
│── Rock/
│── Peace/
│── Stop/
│── Call/
```
Each subfolder corresponds to a gesture label.

## 🚀 Installation & Setup
### 1️⃣ Install Dependencies
```bash
pip install tensorflow numpy matplotlib
```
### 2️⃣ Run Training
```bash
python train.py  # Ensure the script is named train.py
```

## ⚙️ Model Training Details
- **Model Architecture:** Custom CNN with Conv2D, MaxPooling, Flatten, Dense layers.
- **Image Input Size:** 224x224 pixels
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Data Augmentation:** Rotation, Zoom, Flip, Shift

## 🔥 Results & Improvements
- The model improves accuracy with **data augmentation** and **hyperparameter tuning**.
- If performance is unsatisfactory, consider:
  - Collecting **more data** (500+ images per class).
  - Adjusting CNN **depth and filters**.
  - Experimenting with **dropout and batch normalization**.

## 🏆 Next Steps
- Implement **real-time gesture detection** using OpenCV.
- Deploy as a **web app** or **mobile application**.

## 📜 License
This project is open-source under the **MIT License**.

---
🔹 **Developed by:** *[Your Name]*
