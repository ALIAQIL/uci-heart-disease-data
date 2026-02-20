# UCI Heart Disease Classification using PyTorch

![Heart Disease](https://img.shields.io/badge/Dataset-UCI_Heart_Disease-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

## Overview
This repository contains a deep learning project utilizing the **UCI Heart Disease** dataset to predict the presence of heart disease in patients. The project features an end-to-end Machine Learning pipeline encompassing data loading, preprocessing, neural network architecture design with **PyTorch**, training optimization via Early Stopping, and detailed performance evaluation.

## 📊 Dataset
The dataset utilized is the classic **UCI Heart Disease** dataset (`heart_disease_uci.csv`). It consists of multiple clinical and demographic attributes (e.g., age, sex, chest pain type, resting blood pressure, cholesterol levels) used to predict a target outcome indicating if a patient is healthy or suffering from heart disease.

## 🛠️ Technologies & Libraries
- **PyTorch**: For building, training, and predicting with the neural network model.
- **Scikit-learn**: Used for dataset splitting, metric score calculations, and preprocessing.
- **Pandas & NumPy**: For efficient dataset manipulation and numerical computing.
- **Matplotlib & Seaborn**: For plotting training history (Loss & Accuracy curves) and the final Confusion Matrix.

## ✨ Project Highlights
- **Custom Neural Network Formulation:** Defined a custom deep learning classifier (`HeartDiseaseClassifier`) tailored for tabular medical data.
- **Early Stopping:** Integrated a custom EarlyStopping mechanism to halt training when validation metric improvements stagnate, effectively preventing model overfitting.
- **Training Visualization:** Generates side-by-side plots visually mapping the epoch-by-epoch learning phase of both Model Loss and Iterative Accuracy.
- **Detailed Evaluation:** Assesses performance beyond simple accuracy, employing Precision, Recall, F1-Score, and a visually rendered Confusion Matrix in Seaborn.
- **Medical Context Analysis:** Outlines real-world medical implications focusing on the tradeoff between False Positives and False Negatives. In medical diagnostics, minimizing **False Negatives** (predicting a sick patient as healthy) is critical to ensuring no patient is deprived of life-saving interventions.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ALIAQIL/uci-heart-disease-data.git
   cd uci-heart-disease-data
   ```

2. **Install the prerequisites:**
   Create a virtual environment (optional but recommended) and install dependencies.
   ```bash
   pip install torch pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook UCI_Heart_disease.ipynb
   ```
   Execute the cells systematically to observe data preparation, network architecture creation, real-time training loop progression, and standard test-set predictions plotting.

## 📈 Results Summary
The output summarizes the precision metrics and the visual **Confusion Matrix** effectively separating the two target classes:
- **`0`**: Healthy / Pas Malade
- **`1`**: Sick / Malade
