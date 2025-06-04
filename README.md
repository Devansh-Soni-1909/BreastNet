# BreastNet
A neural network-based approach to classify breast cancer.


Breast Cancer Classification with Neural Network

Project Overview

This project implements a Breast Cancer Classification system using a simple neural network built with TensorFlow/Keras. The goal is to classify breast tumors as Benign or Malignant based on features from the Breast Cancer Wisconsin (Diagnostic) Dataset provided by scikit-learn. The model is trained on standardized features, evaluated for accuracy, and includes a predictive system for classifying new data points.

Table of Contents





Project Overview



Dataset



Features



Installation



Usage



Model Architecture



Results



File Structure



Future Improvements



Contributing



License

Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) Dataset from scikit-learn, which contains:





569 samples: 357 Benign (1) and 212 Malignant (0) cases.



30 features: Numerical attributes describing tumor characteristics (e.g., mean radius, texture, perimeter).



Target variable: Binary classification (0 = Malignant, 1 = Benign).

Features





Data Preprocessing: Features are standardized using StandardScaler to ensure consistent scaling.



Neural Network: A simple feedforward neural network with one hidden layer (20 units, ReLU) and an output layer (2 units, sigmoid).



Training: The model is trained for 10 epochs with a validation split of 10%.



Evaluation: Accuracy and loss are visualized for training and validation data, and test accuracy is computed.



Predictive System: Classifies new tumor data as Benign or Malignant based on input features.

Installation

To run this project locally, ensure you have Python 3.6+ installed. Follow these steps:





Clone the repository:

git clone https://github.com/Devansh-Soni-1909/breast-cancer-classification.git
cd breast-cancer-classification



Install the required dependencies:

pip install -r requirements.txt



Create a requirements.txt file with the following:

numpy
pandas
matplotlib
scikit-learn
tensorflow

Usage





Run the main script (breast_cancer_classification.py) to train the model and evaluate its performance:

python breast_cancer_classification.py



The script will:





Load and preprocess the dataset.



Train the neural network.



Plot training/validation accuracy and loss.



Evaluate the model on the test set.



Demonstrate predictions on a sample input.



To use the predictive system, modify the input_data variable in the script with new feature values (30-dimensional vector) and run the script.

Model Architecture

The neural network is a sequential model built with Keras:





Input Layer: Flatten layer to handle 30-dimensional input.



Hidden Layer: Dense layer with 20 units and ReLU activation.



Output Layer: Dense layer with 2 units and sigmoid activation for binary classification.



Optimizer: Adam.



Loss Function: Sparse categorical crossentropy.



Metric: Accuracy.

Results





Training: The model achieves high accuracy on the training and validation sets after 10 epochs.



Test Accuracy: Evaluated on 20% of the dataset (114 samples).



Visualization: Plots of accuracy and loss over epochs are generated to assess model performance.



Prediction: The system correctly classifies new tumor data as Benign or Malignant based on feature inputs.

File Structure

breast-cancer-classification/
│
├── breast_cancer_classification.py  # Main script for data processing, model training, and prediction
├── README.md                       # Project documentation
├── requirements.txt                # List of dependencies
└── plots/                          # Directory for saving accuracy/loss plots (optional)

Future Improvements





Hyperparameter Tuning: Experiment with different network architectures, learning rates, or epochs.



Cross-Validation: Implement k-fold cross-validation for more robust evaluation.



Feature Selection: Use feature importance analysis to reduce dimensionality.



Advanced Models: Explore deeper neural networks, ensemble methods, or other algorithms like XGBoost.



Model Interpretability: Integrate tools like SHAP or LIME to explain predictions.

Contributing

Contributions are welcome! To contribute:





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Make your changes and commit (git commit -m "Add feature").



Push to the branch (git push origin feature-branch).



Open a pull request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

License

This project is licensed under the MIT License. See the LICENSE file for details.
