---
title: RidgeVision AI
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# 🩸 RidgeVision AI — Fingerprint-Based Blood Group Prediction

<div align="center">

![RidgeVision AI](https://img.shields.io/badge/RidgeVision-AI-darkgreen?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Deployed-yellow?style=for-the-badge&logo=huggingface)
![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-~89.5--90%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey?style=for-the-badge)

**RidgeAttenFusionNet** — A dual-branch deep learning ensemble for non-invasive ABO/Rh blood group estimation from fingerprint dermatoglyphics, with Grad-CAM explainability.

# 🔗 **Live Demo:** [(RidgeVision AI)]([[https://sravaninanubala-ridgevision-ai.hf.space/](https://ridgevision-ai.onrender.com/](https://ridgevision-ai.onrender.com)))

📂 **Repository:** [https://github.com/NanubalaSravani/Ridgevision-ai](https://github.com/NanubalaSravani/Ridgevision-ai)

> ⚠️ **Research Prototype v1.0** — Not intended for medical diagnosis or clinical use.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Novelty](#-novelty)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Workflow Pipeline](#-workflow-pipeline)
- [Model Details](#-model-details)
- [Performance Results](#-performance-results)
- [Confusion Matrix](#-confusion-matrix)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Demo Screenshots](#-demo-screenshots)
- [Future Work](#-future-work)
- [Team](#-team)
- [References](#-references)

---

## 🔬 Overview

**RidgeVision AI** is an academic research prototype that explores the correlation between fingerprint ridge patterns (dermatoglyphics) and ABO/Rh blood group classification. The system implements **RidgeAttenFusionNet**, a custom dual-branch ensemble neural network that fuses:

- **Branch 1:** Deep CNN features via transfer learning (EfficientNetB0 / MobileNetV2 with CBAM Attention)
- **Branch 2:** Handcrafted texture features — **LBP** (Local Binary Pattern) + **GLCM** (Gray-Level Co-occurrence Matrix) + **Ridge Density**

Predictions are accompanied by **Grad-CAM** heatmaps for visual explainability, enabling researchers to understand which fingerprint regions drive each prediction.

The ensemble achieves **~89.5%–90% test accuracy** (Kaggle test set evaluation) across all 8 ABO/Rh blood group classes (A+, A−, B+, B−, AB+, AB−, O+, O−).

---

## 💡 Novelty

RidgeAttenFusionNet introduces several distinctive contributions that differentiate it from prior work in fingerprint-based biometric classification:

| # | Novel Contribution | Description |
|:-:|:------------------|:------------|
| 1 | 🔀 **Dual-Branch Hybrid Fusion** | Uniquely combines deep CNN embeddings (EfficientNetB0) with handcrafted texture descriptors (LBP + GLCM + Ridge Density) in a single ensemble — most prior work uses either deep features or handcrafted features, not both |
| 2 | 🎯 **CBAM Attention on Dermatoglyphics** | First application of Convolutional Block Attention Module (channel + spatial) specifically tuned for fingerprint ridge pattern analysis, enabling the network to focus on biologically relevant ridge regions |
| 3 | 🧬 **8-Class ABO/Rh Classification** | Extends beyond binary or 4-class ABO systems to full 8-class ABO/Rh prediction (A+, A−, B+, B−, AB+, AB−, O+, O−) — a significantly harder problem rarely addressed in existing literature |
| 4 | 🔥 **Grad-CAM Explainability for Blood Group Inference** | Integrates gradient-weighted class activation maps to visually explain *which fingerprint regions* drive each blood group prediction, making the system interpretable for research validation |
| 5 | 🧹 **Domain-Specific Preprocessing Pipeline** | Custom CLAHE + Gabor filter bank (8 orientations × 4 frequencies) preprocessing chain designed specifically for ridge enhancement in fingerprint dermatoglyphics, improving feature quality before model inference |
| 6 | ⚖️ **Two-Model Weighted Ensemble** | Trains two architecturally distinct CNN variants (Model-88 and Model-91 style) and combines their softmax outputs, achieving ~89.5–90% ensemble accuracy — outperforming each individual model |
| 7 | 🌐 **End-to-End Web Deployment** | Delivers the full pipeline — from raw fingerprint upload to Grad-CAM-annotated prediction — as a live, publicly accessible web application (HemaPulse AI on Hugging Face Spaces), bridging research and usability |

---

## ✨ Features

| Feature | Description |
|:--------|:------------|
| 🔍 **Non-Invasive Prediction** | Blood group inference from fingerprint images — no blood sample needed |
| 🧠 **Dual-Branch Ensemble** | Combines deep CNN features with handcrafted texture descriptors |
| 🎯 **8-Class Classification** | Supports all ABO/Rh blood groups: A+, A−, B+, B−, AB+, AB−, O+, O− |
| 🔥 **Grad-CAM Explainability** | Visual heatmap overlay showing which regions influence predictions |
| 📊 **Confidence Matrix** | Class-wise probability distribution for all 8 groups |
| 🧹 **Preprocessing Pipeline** | CLAHE + Gabor filtering + denoising + ridge enhancement |
| 🌐 **Web Interface** | Clean HemaPulse AI-style frontend deployed on Hugging Face Spaces |
| 📈 **~89.5%–90% Ensemble Accuracy** | Macro-average precision, recall, and F1 score all at 0.90 (Kaggle test set evaluation) |

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph BRANCH1["🧠 Branch 1 — CNN"]
        A1["Input Fingerprint Image"] --> B1["CLAHE Preprocessing"]
        B1 --> C1["Gabor Filter Enhancement"]
        C1 --> D1["EfficientNetB0 Backbone<br/>(Pretrained ImageNet)"]
        D1 --> E1["CBAM Attention Module<br/>(Channel + Spatial)"]
        E1 --> F1["Global Average Pooling"]
        F1 --> G1["Dense (512 → 256)"]
        G1 --> H1["Dropout (0.4)"]
        H1 --> I1["Softmax Output (8 classes)"]
    end

    subgraph BRANCH2["🔬 Branch 2 — Texture"]
        A2["Input Fingerprint Image"] --> B2["CLAHE + Denoising"]
        B2 --> C2["Ridge Enhancement (Gabor)"]
        C2 --> D2["LBP Feature Extraction<br/>GLCM Texture Descriptors<br/>Ridge Density Map"]
        D2 --> E2["MLP Feature Fusion"]
        E2 --> F2["Dense Layer (256 → 128)"]
        F2 --> G2["BatchNorm + Dropout (0.3)"]
        G2 --> H2["Softmax Output (8 classes)"]
    end

    I1 & H2 --> ENS["⚖️ Ensemble Averaging<br/>(Weighted Average of Branch 1 + Branch 2 probabilities)"]
    ENS --> OUT["🩸 Final Predicted Blood Group (ABO/Rh)<br/>+ Confidence Score + Grad-CAM Heatmap"]
```

---

## 🔄 Workflow Pipeline

```mermaid
flowchart LR
    A["📤 User Submits Fingerprint<br/>Upload PNG/JPG/BMP<br/>to web interface"] --> B["⚙️ Image is Processed<br/>Base64 decode<br/>+ preparation for pipeline"]
    B --> C["🔍 Key Features Identified<br/>LBP, GLCM, Ridge Density<br/>& Texture features extracted"]
    C --> D["🎨 Image is Enhanced<br/>CLAHE + Denoising<br/>+ Ridge Enhancement applied"]
    D --> E["🤖 Model Predicts Blood Group<br/>Pre-trained Keras ensemble<br/>runs inference on processed image"]
    E --> F["📊 System Outputs Blood Group<br/>A+, A−, B+, B−,<br/>AB+, AB−, O+, or O−"]
    F --> G["🔥 Visualizes Important Regions<br/>Grad-CAM heatmap displayed"]
    F --> H["📈 Confidence Quantified<br/>Probability matrix +<br/>class-wise scores shown"]
    G & H --> I["✅ Results Presented to User<br/>Results + Confidence<br/>+ Heatmap shown"]
```

---

## 🧠 Model Details

### RidgeAttenFusionNet — Two-Model Ensemble

> **RidgeAttenFusionNet** is a custom-designed experimental ensemble architecture developed for this research.

The system trains **two independent CNN models** (Model 88-style and Model 91-style) and averages their softmax outputs:

| Component | Specification |
|:----------|:-------------|
| **Backbone** | EfficientNetB0 (pretrained on ImageNet) |
| **Attention** | CBAM — Channel Attention + Spatial Attention |
| **Texture Features** | LBP (radius=1, 8 neighbors) + GLCM (contrast, correlation, energy, homogeneity) |
| **Ridge Features** | Gabor filter bank (8 orientations) for ridge density |
| **Image Preprocessing** | CLAHE (clip limit 2.0, tile grid 8×8) + Gaussian denoising |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output Classes** | 8 (A+, A−, AB+, AB−, B+, B−, O+, O−) |
| **Loss Function** | Categorical Cross-Entropy |
| **Optimizer** | Adam (lr=1e-4, with ReduceLROnPlateau) |
| **Regularization** | Dropout (0.3–0.4) + BatchNormalization |
| **Training Platform** | Kaggle (GPU P100) |

### CBAM Attention Module

```mermaid
flowchart LR
    A["Input Feature Map"] --> B["Channel Attention\nGlobal Avg Pool + Global Max Pool → MLP → Sigmoid → Scale"]
    B --> C["Spatial Attention\nChannel-wise Avg + Max → Conv2D(7×7) → Sigmoid → Scale"]
    C --> D["Attended Feature Map"]
```

### Preprocessing Pipeline

```mermaid
flowchart TD
    A["🖼️ Raw Fingerprint Image"] --> B["Grayscale Convert"]
    B --> C["CLAHE Equalization\n(Contrast Limited Adaptive Histogram Equalization)"]
    C --> D["Gaussian Denoising"]
    D --> E["Gabor Ridge Enhancement\n(8 orientations × 4 frequencies)"]
    E --> F["Resize to 224×224"]
    F --> G["Normalize [0, 1]"]
    G --> H["✅ Ready for Model Inference"]
```

## 📌 Experimental Note

This system is designed for research purposes to explore potential biometric correlations using deep learning. The results are statistical predictions and should be interpreted as experimental findings rather than deterministic biological conclusions.

---

## 📊 Performance Results

### Ensemble Accuracy: **~89.5%–90%** (Kaggle test set evaluation)

| Model | Test Accuracy |
|:------|:-------------|
| Model 88-style | 86.0% |
| Model 91-style | 89.4% |
| **Ensemble (Avg)** | **89.5–90.0%** |

### Per-Class Classification Report (1200 test samples, 150 per class)

| Blood Group | Precision | Recall | F1-Score | Support |
|:-----------|:---------:|:------:|:--------:|:-------:|
| **A+** | 0.88 | 0.89 | 0.88 | 150 |
| **A−** | 0.92 | 0.89 | 0.91 | 150 |
| **AB+** | 0.88 | 0.86 | 0.87 | 150 |
| **AB−** | 0.92 | 0.92 | 0.92 | 150 |
| **B+** | 0.89 | 0.92 | 0.90 | 150 |
| **B−** | 0.92 | 0.92 | 0.92 | 150 |
| **O+** | 0.93 | 0.85 | 0.89 | 150 |
| **O−** | 0.83 | 0.91 | 0.87 | 150 |
| **Accuracy** | — | — | **0.90** | **1200** |
| **Macro Avg** | 0.90 | 0.90 | 0.90 | 1200 |
| **Weighted Avg** | 0.90 | 0.90 | 0.90 | 1200 |

---

## 🗂️ Confusion Matrix

The confusion matrix below shows the RidgeVision Retrained Two-Model Ensemble performance across all 8 blood group classes on the test set:

| Actual ↓ \ Predicted → | **A+** | **A−** | **AB+** | **AB−** | **B+** | **B−** | **O+** | **O−** |
|:-----------------------|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|:------:|
| **A+** | **134** | 0 | 5 | 0 | 0 | 0 | 3 | 8 |
| **A−** | 0 | **134** | 2 | 2 | 1 | 6 | 3 | 2 |
| **AB+** | 6 | 0 | **129** | 0 | 8 | 0 | 3 | 4 |
| **AB−** | 0 | 2 | 0 | **138** | 2 | 4 | 0 | 4 |
| **B+** | 0 | 2 | 5 | 3 | **138** | 2 | 0 | 0 |
| **B−** | 0 | 3 | 0 | 4 | 5 | **138** | 0 | 0 |
| **O+** | 7 | 4 | 1 | 1 | 0 | 0 | **127** | 10 |
| **O−** | 6 | 1 | 4 | 2 | 1 | 0 | 0 | **136** |

**Key Observations:**
- **AB−** and **B−** achieve the highest per-class accuracy (138/150 correct)
- **O+** shows occasional confusion with A+ and O−
- **A+** has minor confusion with O− (8 misclassifications), which is a known challenge in dermatoglyphic classification
- Overall strong diagonal dominance confirms excellent model performance

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Backend Framework** | FastAPI | API endpoints, inference handling, and backend logic |
| **Frontend** | HTML5, CSS3, JavaScript | User interface for fingerprint upload and results display |
| **Static Hosting** | FastAPI StaticFiles | Serves frontend assets |
| **Deep Learning** | TensorFlow / Keras | Model training and inference |
| **Computer Vision** | OpenCV, scikit-image | Image preprocessing and feature extraction |
| **Explainability** | Grad-CAM | Visual explanation of model predictions |
| **Training Environment** | Kaggle (GPU P100) | Model training and experimentation |
| **Deployment** | Hugging Face Spaces | Cloud hosting and live demo |

---

## 📁 Project Structure

```
Ridgevision-ai/
│
├── app.py                          # Main application entry point (FastAPI)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── model_88/                   # Style-88 trained weights (.h5 / .keras)
│   ├── model_91/                   # Style-91 trained weights (.h5 / .keras)
│   └── ensemble_config.json        # Ensemble weight configuration
│
├── preprocessing/
│   ├── clahe_pipeline.py           # CLAHE + Gaussian denoising
│   ├── gabor_enhance.py            # Gabor filter ridge enhancement
│   └── feature_extraction.py       # LBP + GLCM + Ridge density
│
├── model_architecture/
│   ├── ridgeatten_fusion.py        # RidgeAttenFusionNet definition
│   ├── cbam_attention.py           # CBAM Channel + Spatial Attention
│   └── ensemble_predict.py         # Ensemble averaging + inference
│
├── explainability/
│   └── gradcam.py                  # Grad-CAM heatmap generation
│
├── frontend/
│   ├── templates/
│   │   └── index.html              # Main web interface (HemaPulse AI UI)
│   └── static/
│       ├── style.css               # Styling
│       └── script.js               # Frontend logic (drag-drop, fetch)
│
├── notebooks/
│   └── 90_accuracy_training.ipynb  # Kaggle training notebook
│
├── assets/
│   ├── confusion_matrix.png        # Ensemble confusion matrix
│   ├── workflow_diagram.png        # Pipeline flowchart
│   └── classification_report.txt  # Full test metrics
│
└── report_images/
    └── ridgevision_train_both_ensemble_report.txt
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8–3.11
- pip
- (Recommended) NVIDIA GPU with CUDA 11.x for fast inference

### 1. Clone the Repository

```bash
git clone https://github.com/NanubalaSravani/Ridgevision-ai.git
cd Ridgevision-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core requirements:**
```
tensorflow>=2.10
keras
numpy
pandas
scikit-learn
scikit-image
opencv-python
matplotlib
seaborn
fastapi
uvicorn
Pillow
```

### 4. Download Pre-trained Models

Place model weights in the `models/` directory. If using Kaggle-trained weights:

```bash
# From Kaggle (requires kaggle API token)
kaggle kernels output sravaninanubala/90-accuracy -p ./models/
```

Or download directly from the Hugging Face repo and place `.h5` / `.keras` files in `models/`.

---

## 🚀 Running the App

### Local (FastAPI)

```bash
uvicorn app:app --reload
```
Open your browser at: `http://localhost:8000`

### Using the Web Interface

1. **Upload** a fingerprint image (PNG, JPG, or BMP)
2. The system automatically **validates image quality** (clarity + completeness)
3. Click **Analyze** — the AI inference engine runs:
   - CLAHE + Gabor preprocessing
   - LBP / GLCM feature extraction
   - Dual-branch ensemble inference
   - Grad-CAM heatmap generation
4. **Results** displayed:
   - Predicted blood group (e.g., **AB−**)
   - Confidence percentage
   - Class likelihood matrix (all 8 probabilities)
   - Grad-CAM attention overlay

---

## 🖥️ Demo Screenshots

### Web Interface — RidgeVision AI (HemaPulse Frontend)

The deployed interface at [sravaninanubala-ridgevision-ai.hf.space](https://sravaninanubala-ridgevision-ai.hf.space/) features:

**Step 1 — Fingerprint Scanner Panel**

![RidgeVision AI — Fingerprint Scanner](screenshot_scanner.png)<img width="1920" height="1140" alt="screenshot_scanner" src="https://github.com/user-attachments/assets/fcc14b43-3887-49bc-8cd3-0f81dbe81695" />


**Step 2 — Diagnostic Report Panel**

![RidgeVision AI — Diagnostic Report](screenshot_results.png)<img width="1920" height="1140" alt="screenshot_results" src="https://github.com/user-attachments/assets/423db027-d19c-45a1-b353-2bf2bff2bc53" />



---

## 🔮 Future Work

| Enhancement | Description |
|:-----------|:------------|
| 🗃️ **Larger Dataset** | Train on a more diverse, larger fingerprint-blood group dataset for better generalization |
| 🔍 **Multi-finger Fusion** | Use all 10 fingerprints to improve prediction reliability |
| 📱 **Mobile App** | Deploy as a lightweight mobile application (TFLite) |
| 🩺 **Clinical Validation** | Partner with medical institutions to validate predictions on clinical samples |
| ⚡ **Model Optimization** | Quantization + pruning for edge device deployment |
| 🌐 **Multilingual UI** | Add Telugu and Hindi language support |
| 📚 **Expanded Classes** | Extend to include MN, P, Lewis blood group systems |
| 🔒 **Privacy-Preserving ML** | Federated learning for sensitive biometric data |

---

## 👥 Team

| Name |
|:-----|
| **N. Sravani** |
| **P. Likhitha** |

---

## ⚠️ Disclaimer

> This project is an **academic research prototype** developed as a B.Tech mini-project. It demonstrates experimental correlations between fingerprint ridge patterns and blood group classification using machine learning. It is **NOT clinically validated** and must **not be used for medical diagnosis**. Blood group determination must always be performed by a certified medical laboratory.

---

## 📚 References

1. Maltoni, D., Maio, D., Jain, A. K., & Prabhakar, S. (2009). *Handbook of Fingerprint Recognition*. Springer.
2. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
4. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. *IEEE TPAMI*.
5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
6. Cummins, H., & Midlo, C. (1943). *Finger Prints, Palms and Soles*. Philadelphia: Blakiston.
7. Prasanth Vaidya S., Tyagi A., Ashok S., Naga Satish G., & Meenakshi R. (2024). Advancements in hemotype identification: Fingerprint analysis for blood group determination. *Department of CSE, BVRIT HYDERABAD College of Engineering for Women; Alliance University; Chennai Institute of Technology*.

---

## 📄 License

This project is released for **academic research and educational purposes only**.
© 2025–2026 N. Sravani & Team — The Apollo University.

---

<div align="center">

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow?style=for-the-badge)](https://sravaninanubala-ridgevision-ai.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/NanubalaSravani/Ridgevision-ai)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training%20Notebook-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/sravaninanubala/90-accuracy)

</div>
