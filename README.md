# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

Credit Card Fraud Detection is a Machine Learning project that identifies whether a credit card transaction is **fraudulent** or **genuine**. Since fraudulent transactions are extremely rare compared to legitimate ones, this project focuses on handling **class imbalance** using **SMOTE (Synthetic Minority Oversampling Technique)** and compares the performance of two machine learning models: **Logistic Regression** and **Random Forest Classifier**.

This project demonstrates a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), feature scaling, model training, evaluation, and performance comparison.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions using Machine Learning.
* Perform data preprocessing and feature engineering.
* Handle class imbalance using SMOTE.
* Train multiple classification models.
* Evaluate model performance using Precision, Recall, F1-Score, Accuracy, and Confusion Matrix.
* Compare model performance and identify the best-performing model.

---

## 📂 Dataset

**Dataset Name:** `creditcard.csv`

The dataset contains anonymized credit card transactions made by European cardholders.

### Dataset Features

* **Time** – Time elapsed between transactions.
* **V1 – V28** – PCA transformed numerical features.
* **Amount** – Transaction amount.
* **Class** – Target Variable

  * **0 → Genuine Transaction**
  * **1 → Fraudulent Transaction**

---

## 🛠️ Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

## 📚 Machine Learning Workflow

### 1. Import Libraries

Imported all required Python libraries for data manipulation, visualization, preprocessing, model building, and evaluation.

---

### 2. Load Dataset

Loaded the credit card transaction dataset using Pandas.
The Dataset File is too long to Upload so you can download it directly from kaggle 
```python
df = pd.read_csv("creditcard.csv")
```

---

### 3. Data Exploration

Performed initial data analysis using:

* `head()`
* `shape`
* `info()`
* `describe()`
* Missing value detection

---

### 4. Data Visualization

Created visualizations to understand the dataset:

* Class Distribution Plot
* Correlation Heatmap

---

### 5. Data Preprocessing

Performed the following preprocessing steps:

* Removed missing values
* Standardized the **Amount** feature using **StandardScaler**
* Removed the **Time** column
* Split features and target variable

---

### 6. Train-Test Split

Dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

using stratified sampling to preserve class distribution.

---

### 7. Handle Class Imbalance

Applied **SMOTE** to generate synthetic fraud samples and balance the training dataset.

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

---

### 8. Model Training

The following Machine Learning algorithms were trained:

* Logistic Regression
* Random Forest Classifier

---

### 9. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

### 10. Model Comparison

A comparison table was created to compare:

* Accuracy
* Precision
* Recall
* F1 Score

This helps identify the best-performing model.

---

## 📈 Results

The trained models successfully classified credit card transactions into **Fraudulent** and **Genuine** categories.

The project demonstrates that:

* Proper preprocessing significantly improves performance.
* SMOTE effectively handles class imbalance.
* Random Forest generally performs better than Logistic Regression for this problem.

---

## 📊 Evaluation Metrics

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Classification Report
* Confusion Matrix

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/

│── Credit_Card_Fraud_Detection.ipynb
│── creditcard.csv
│── README.md
│── requirements.txt
│── images/
│     ├── class_distribution.png
│     ├── correlation_heatmap.png
│     ├── confusion_matrix_lr.png
│     ├── confusion_matrix_rf.png
│     └── model_comparison.png
```

---

## ▶️ How to Run the Project

### Step 1

Clone the repository.

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

---

### Step 2

Navigate to the project folder.

```bash
cd Credit-Card-Fraud-Detection
```

---

### Step 3

Install the required libraries.

```bash
pip install -r requirements.txt
```

---

### Step 4

Open the notebook.

jupyter notebook Credit_Card_Fraud_Detection.ipynb

---

## 📌 Future Improvements

* XGBoost Classifier
* LightGBM
* CatBoost
* Hyperparameter Tuning
* ROC-AUC Curve Analysis
* Precision-Recall Curve
* Feature Importance Analysis
* Streamlit Web Application
* Real-time Fraud Detection API

---

## 🎓 Learning Outcomes

Through this project, I learned:

* Data preprocessing techniques
* Feature scaling
* Handling imbalanced datasets using SMOTE
* Classification algorithms
* Model evaluation metrics
* Fraud detection workflow
* End-to-end Machine Learning pipeline

---

## 👨‍💻 Author

**Shashank Singh Chauhan**

**Data Analyst | Machine Learning Enthusiast | Python Developer**

If you found this project useful, feel free to ⭐ this repository and connect with me on LinkedIn.

---

## 📜 License

This project is intended for educational and learning purposes.
