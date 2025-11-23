# 📘 K-Nearest Neighbors (KNN) Diabetes Prediction Project

## 📌 **Project Overview**

This project implements a **K-Nearest Neighbors (KNN)** classification model to predict diabetes based on patient health attributes. The workflow includes:

* Data loading
* Data cleaning
* Feature scaling
* Splitting into train/test sets
* Finding the best K value
* Training the KNN model
* Evaluating performance metrics
* Saving the trained model pipeline
* Visualizing accuracy vs K value

The model is saved as a reusable pipeline using `joblib`.

---

## 📂 **Project Structure**

```
Project Folder/
│
├── diabetes.csv                # Dataset file
├── diabities.py                # Main Python script with full pipeline
├── artifacts_sample/           # Folder where the trained model is saved
│   └── KNN_pipeline.joblib     # Saved KNN model pipeline
└── README.md                   # Documentation
```

---

## 🧪 **Required Libraries**

Make sure these packages are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 🚀 **How to Run the Project**

### **Step 1: Ensure diabetes.csv is in the same folder**

Make sure the dataset file exists at:

```
ProjectFolder/diabetes.csv
```

### **Step 2: Run the script**

Open terminal in VS Code or CMD:

```bash
python diabities.py
```

---

## 🔍 **What the Script Does**

### ✔ Loads and cleans the dataset

Removes missing values and separates **features (X)** and **target (y)**.

### ✔ Scales the features

Uses **StandardScaler** for normalization.

### ✔ Splits data

Divides the dataset into **training and testing sets**.

### ✔ Finds the best K

Evaluates accuracy for K = 1 to 24 and chooses the best.

### ✔ Trains the KNN model

Builds both:

* A raw KNN model
* A full **pipeline** with scaling + KNN

### ✔ Saves the model

The pipeline is saved to:

```
artifacts_sample/KNN_pipeline.joblib
```

### ✔ Evaluates model performance

Prints:

* Accuracy
* Confusion Matrix
* Classification Report
* ROC-AUC Score
* Precision, Recall, F1 Score

### ✔ Plots accuracy vs K value

Displays graph of model accuracy compared to different K values.

---

## 📊 **Sample Output Summary (Formatted)**

```
Best K Value: 18
Accuracy (KNN Pipeline, k=18): 75.97%
Final Accuracy: 76.62%

Confusion Matrix:
[[89 10]
 [26 29]]

ROC-AUC Score: 0.7131
Precision: 0.7436
Recall: 0.5272
F1 Score: 0.6170
```

---

## 💾 **Model Saving Location**

The trained KNN model is saved automatically here:

```
artifacts_sample/KNN_pipeline.joblib
```

You can load it later using:

```python
model = joblib.load("artifacts_sample/KNN_pipeline.joblib")
```

---

## 🎯 **Purpose of the Project**

This project demonstrates:

* Building an ML pipeline
* Data preprocessing
* Hyperparameter tuning (best K)
* Model evaluation
* Saving reusable ML models

It is suitable for:

* College mini-projects
* Machine Learning beginners
* Model comparison experiments

---

## 🛠 **Future Improvements**

You can enhance the project by adding:

* Cross-validation
* Grid search for K
* GUI with Tkinter/Streamlit
* API using FastAPI/Flask
* More models (SVM / Logistic Regression)

---

## 📧 **Support**

If you want:

* A report for this project
* PPT for seminar
* GUI/API
* Explanation of each function

Just ask — I will prepare it for you! 😊
