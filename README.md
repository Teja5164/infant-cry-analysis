# Multimodal Infant Cry Analysis System

## Overview
This repository contains the ongoing development of a **Multimodal Infant Cry Analysis Machine Learning System** designed to classify infant cry types and provide explainable reasoning using multiple data modalities.

The project is being developed as part of a technical evaluation and is structured to be **scalable, explainable, and production-ready**.

---

## Project Objective
To build a multimodal ML model that predicts infant cry types such as:
- Hunger
- Pain
- Discomfort
- Sleepy
- Sick / Illness

**Inputs**
- Cry Audio (WAV, 16 kHz)
- Baby Images (Facial / Visual cues)
- Vitals (Heart Rate, Respiration, Temperature, SpO₂)

**Outputs**
- Cry Type Classification
- Explainable Reasoning (Model Interpretability)

---

## Current Project Status

### ✅ Completed
#### Audio Modality
- Audio dataset loading and preprocessing
- MFCC feature extraction
- CNN-based audio classification model (PyTorch)
- Class imbalance handling using class weights
- Model training and validation
- **Audio Grad-CAM implemented successfully** for explainability

This confirms that:
- The audio pipeline is stable
- Model explanations are available and interpretable
- The architecture is compatible with future multimodal fusion

---

### ⚠️ Partially Completed / Experimental
#### Audio Optimization Attempts
- Advanced augmentation and cross-validation strategies were explored
- Some experimental configurations did not yield improved accuracy
- These attempts were intentionally rolled back to preserve model stability

This reflects a **controlled and iterative ML development approach**.

---

### ⏳ Pending
#### Image Modality
- Image pipeline structure is created
- Preprocessing and model training scripts are prepared
- **Dataset alignment is pending** due to unavailability of infant image datasets with labels matching the audio classes

#### Vitals Modality
- Planned as a tabular / time-series input
- Will be integrated after image modality alignment

---

## Repository Structure
infant-cry-analysis/
│
├── src/
│ ├── audio/ # Audio preprocessing, training, Grad-CAM
│ ├── image/ # Image pipeline (scaffolded)
│ ├── fusion/ # Multimodal fusion (planned)
│
├── data/
│ ├── audio/
│ ├── images/
│
├── models/ # Saved models (ignored in git)
├── README.md
└── .gitignore


---

## Explainability
- Audio explainability implemented using **Grad-CAM**
- Image explainability planned using CNN-based attention maps
- Multimodal reasoning will combine modality-level explanations

---

## Deployment (Planned)
- Cloud-based inference (AWS)
- Modular architecture suitable for edge / mobile deployment
- Inference latency target: ≤ 3 seconds

---

## Notes
- The project is intentionally committed at a **stable checkpoint**
- Dataset limitations are acknowledged transparently
- The system is designed to be **extended seamlessly** once aligned datasets are available

---

## Next Steps
- Acquire or align infant image datasets
- Complete image model training and explainability
- Add vitals modality
- Implement multimodal fusion network
- Final evaluation on unseen data

---

## Author
Teja Prakash
