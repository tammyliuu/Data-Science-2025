# Data-Science-2025
# Fraud Detection with Imbalanced Class Data

This project tackles the problem of fraud detection using machine learning techniques to handle imbalanced data. The dataset used contains transaction details, where the target variable (`Class`) indicates whether a transaction is fraudulent (1) or non-fraudulent (0). The project applies various sampling techniques to balance the dataset and compares the performance of different models.

## Project Features

- **Dataset**: The project uses a dataset named `creditcard.csv` that contains transaction data.
- **Objective**: Detect fraudulent transactions and handle the imbalanced nature of the dataset where fraud is a rare occurrence.

### Methods Used to Address Imbalance

1. **Oversampling**:
    - **SMOTE (Synthetic Minority Over-sampling Technique)**: Synthetic data points are generated for the minority class (fraud cases).
    - **ADASYN (Adaptive Synthetic Sampling)**: Focuses on generating synthetic data in regions where the minority class is harder to classify.
    - **Bootstrap & Bagging**: Balances the dataset by resampling and creating multiple sub-datasets to reduce the imbalance and improve model performance.

2. **Undersampling**:
    - **Cluster Centroid**: Reduces the majority class by clustering and downsampling, effectively balancing the dataset for training.

### Models Implemented

- **Random Forest Classifier**: An ensemble method that constructs multiple decision trees and combines their results to improve accuracy and prevent overfitting.
- **Bagging Classifier with Decision Trees**: Uses bootstrap sampling to create multiple subsets of the data and trains decision trees on each subset.
- **Logistic Regression**: A linear model trained on a resampled dataset using the Cluster Centroid method.

## Steps Involved

1. **Data Loading and Exploration**:
    - Load the `creditcard.csv` dataset.
    - Explore the imbalance in the dataset:
      - **No Frauds**: Approx. 99.83% of the dataset.
      - **Frauds**: Approx. 0.17% of the dataset.

2. **Data Preprocessing**:
    - Split the dataset into features (X) and target (y).
    - Perform a train-test split to prepare data for model training and evaluation.

3. **Oversampling and Undersampling**:
    - Apply SMOTE and ADASYN to oversample the minority class.
    - Use bootstrap bagging and undersampling methods (Cluster Centroid) to balance the data.

4. **Model Training and Evaluation**:
    - Train the models using different sampling techniques.
    - Plot the class distribution after resampling.
    - Generate classification reports and confusion matrices to compare the performance of each model.

### Results

- **Confusion Matrices**: Plots of confusion matrices after applying different sampling methods.
- **Classification Reports**: Include precision, recall, F1-score, and support for each sampling method (SMOTE, ADASYN, Bagging, and Cluster Centroid).

## Libraries Used

- `pandas`: For data manipulation and analysis.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms, data splitting, and evaluation metrics.
- `imblearn`: For handling imbalanced datasets (oversampling and undersampling techniques).

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/your-repo/fraud-detection.git

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
3. Run the Python script:

    ```bash
    python ds.py

### Conclusion
This project demonstrates how to handle imbalanced data in fraud detection using various oversampling and undersampling techniques. The results show the effectiveness of each method in improving the classification of fraudulent transactions.