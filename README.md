# 🧠 TumorVision
### Brain Tumor Detection using Deep Learning (VGG16 + Streamlit)

TumorVision is a deep learning–based web application that detects the presence of brain tumors from MRI images.  
The system uses a **pre-trained VGG16 model** and provides a **simple Streamlit interface** for fast and accurate predictions.

---

## 🚀 Demo

### 🔹 Application Interface
![App Interface](assets/Screenshot%202025-12-18%20002200.png)


### 🔹 Prediction Result
![Prediction Result](assets/Screenshot%202025-12-18%20002431.png)

---

## ✨ Features

- 🧠 Brain tumor detection from MRI images  
- ⚡ Fast and accurate predictions  
- 🎯 Transfer learning with **VGG16**
- 🖥️ Interactive **Streamlit** web app  
- 📊 Confidence score with prediction  
- 🧪 Image preprocessing & normalization  

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **VGG16 (Transfer Learning)**
- **Streamlit**
- **NumPy, OpenCV, PIL**
- **Pickle**

---

## 📂 Project Structure

```text
TumorVision/
│
├── app.py                  # Streamlit application
├── Brain Tumor.ipynb       # Model training & experiments
├── utils.py                # Helper functions
├── requirements.txt        # Project dependencies
├── assets/                 # UI & result screenshots
└── README.md               # Project documentation
```

## ⚙️ Getting Started

Follow the steps below to run the project locally.

### 1️⃣ Create a Virtual Environment
```bash
conda create -p venv python=3.10.6 -y
```

### 2️⃣ Activate the Environment
```bash
conda activate venv/
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
🎉 Open your browser and upload an MRI image to get predictions.




## 🧠 About the Project

### 🔍 Problem Statement

Brain tumor detection is a critical task in medical imaging. Manual diagnosis is often time-consuming and challenging due to the complex structure of the human brain and the variability in tumor appearance.  
There is a strong need for an **automated and accurate system** to assist healthcare professionals in diagnosis.

---

### 🎯 Project Objective

The primary objectives of this project are to:

- Automatically classify MRI brain scans as:
  - **Tumor**
  - **No Tumor**
- Handle variations in MRI images
- Provide a **user-friendly interface**
- Deliver **quick and reliable predictions**

---

## 🤖 Machine Learning Model

To achieve accurate classification, the project uses a **pre-trained VGG16 convolutional neural network**.

- Transfer learning applied for better performance on limited data
- Data augmentation techniques used:
  - Rotation
  - Flipping
  - Scaling
  - Shifting
- Regularization methods implemented to reduce overfitting
- Model trained and evaluated on MRI images
- Final trained model saved using **pickle**

---

## 🧬 About VGG16

VGG16 is a deep convolutional neural network widely used for image classification tasks.

### Working Pipeline

1. MRI Image Input  
2. Convolutional layers for feature extraction  
3. Pooling layers for dimensionality reduction  
4. Flattening of extracted features  
5. Fully connected layers for classification  
6. Output: **Tumor / No Tumor** with confidence score  

---

## 🖥️ Streamlit Web Application

The Streamlit-based interface allows users to:

- Upload MRI brain images
- Get real-time predictions
- View classification results with confidence
- Use a clean, minimal, and responsive UI

No additional backend or frontend setup is required.

---

## 🌱 Future Scope

- Multi-class tumor classification  
- Tumor localization and segmentation  
- Explainable AI techniques (Grad-CAM)  
- Cloud deployment (AWS / GCP)  
- Medical-grade validation and testing  

