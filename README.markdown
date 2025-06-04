# 🌺 BreastNet: Breast Cancer Classification 🌺

---

Welcome to **BreastNet**, a beautifully crafted project for classifying breast tumors as **Benign** or **Malignant** using a neural network built with **TensorFlow/Keras**. Powered by the **Breast Cancer Wisconsin (Diagnostic) Dataset**, this Google Colab notebook offers a streamlined pipeline for data preprocessing, model training, evaluation, and prediction, making it accessible for both learning and real-world applications.

---

## ✨ Project Overview

BreastNet leverages a simple feedforward neural network to predict tumor malignancy based on 30 features from the Breast Cancer Wisconsin Dataset. The notebook includes data preprocessing, model training, performance visualization, and a predictive system for classifying new tumor data. Designed for ease of use in Google Colab, it’s perfect for researchers, students, and enthusiasts exploring machine learning in healthcare.

---

## 📋 Table of Contents

- [🌺 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🛠 Features](#-features)
- [🚀 Getting Started](#-getting-started)
- [🎯 Usage](#-usage)
- [🧠 Model Architecture](#-model-architecture)
- [📈 Results](#-results)
- [📂 Repository Structure](#-repository-structure)
- [🌟 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** from scikit-learn, featuring:

- **569 samples**: 357 Benign (1) and 212 Malignant (0) cases.
- **30 features**: Numerical attributes (e.g., mean radius, texture, perimeter) derived from digitized images of fine needle aspirates (FNA).
- **Target variable**: Binary classification (0 = Malignant, 1 = Benign).

---

## 🛠 Features

- **Data Preprocessing**: Standardizes features using `StandardScaler` for consistent scaling.
- **Neural Network**: A clean feedforward network with one hidden layer (20 units, ReLU) and an output layer (2 units, sigmoid).
- **Training**: Trains for 10 epochs with a 10% validation split for robust performance.
- **Evaluation**: Visualizes accuracy and loss curves using Matplotlib and computes test accuracy.
- **Predictive System**: Classifies new tumor data as Benign or Malignant with a user-friendly interface.

---

## 🚀 Getting Started

Since this project is a Google Colab notebook, no local setup is required! All dependencies (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`) are pre-installed in Colab’s environment.

1. **Access the Notebook**:
   - Visit the GitHub repository: [BreastNet](https://github.com/Devansh-Soni-1909/BreastNet)
   - Open the notebook: `BreastNet_Breast_Cancer_Detection_with_Neural_Networks.ipynb`

2. **Run in Google Colab**:
   - Click the **Open in Colab** badge on the GitHub repository (or copy the notebook URL).
   - Alternatively, download the `.ipynb` file and upload it to [Google Colab](https://colab.research.google.com/).
   - Colab will automatically handle all dependencies.

3. **Prerequisites**:
   - A web browser and Google account to access Colab.
   - No local Python installation or dependency management is needed.

---

## 🎯 Usage

1. **Open the Notebook**:
   - Load `BreastNet_Breast_Cancer_Detection_with_Neural_Networks.ipynb` in Google Colab.
   - Run all cells sequentially by clicking **Run All** (or `Ctrl+F9`) in Colab.

2. **What It Does**:
   - Loads and preprocesses the Breast Cancer Wisconsin Dataset.
   - Trains a neural network for 10 epochs.
   - Generates plots for training/validation accuracy and loss.
   - Evaluates the model on a test set (20% of data).
   - Demonstrates predictions using a sample 30-feature input.

3. **Make Predictions**:
   - Modify the `input_data` variable in the notebook’s predictive system section with a new 30-feature vector.
   - Run the prediction cell to classify the tumor as **Benign** or **Malignant**.

---

## 🧠 Model Architecture

The neural network is a sleek, sequential model built with Keras:

- **Input Layer**: `Flatten` layer to process the 30-dimensional input.
- **Hidden Layer**: `Dense` layer with 20 units and ReLU activation for feature extraction.
- **Output Layer**: `Dense` layer with 2 units and sigmoid activation for binary classification.
- **Optimizer**: Adam for efficient optimization.
- **Loss Function**: Sparse categorical crossentropy for binary classification.
- **Metric**: Accuracy to monitor performance.

---

## 📈 Results

- **Training**: Achieves high accuracy on training and validation sets after 10 epochs.
- **Test Accuracy**: Evaluated on a 20% test split (114 samples).
- **Visualizations**: Elegant plots of accuracy and loss over epochs, displayed in the notebook.
- **Predictions**: Accurately classifies new tumor data as Benign or Malignant.

---

## 📂 Repository Structure

```
BreastNet/
├── BreastNet_Breast_Cancer_Detection_with_Neural_Networks.ipynb  # Colab notebook with full implementation
├── README.md                                                    # This aesthetic documentation
```

---

## 🌟 Future Improvements

- **Hyperparameter Tuning**: Experiment with deeper architectures, learning rates, or more epochs.
- **Cross-Validation**: Add k-fold cross-validation for robust evaluation.
- **Feature Selection**: Analyze feature importance to reduce dimensionality.
- **Advanced Models**: Explore ensemble methods or algorithms like XGBoost.
- **Interpretability**: Integrate SHAP or LIME for explainable predictions.

---

## 🤝 Contributing

We’d love your contributions to enhance BreastNet! To contribute:

1. Fork the repository: [BreastNet](https://github.com/Devansh-Soni-1909/BreastNet).
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request on GitHub.

Please adhere to **PEP 8** style guidelines and include clear documentation.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

🌟 **Thank you for exploring BreastNet! Together, let’s advance breast cancer detection with the power of AI.** 🌟