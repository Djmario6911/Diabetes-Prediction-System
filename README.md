# 🩺 Diabetes Prediction Using Machine Learning

A machine learning project that predicts whether a patient is likely to have diabetes based on medical and demographic information.

This project was developed as part of an **AI/ML Internship at Zynvex Solutions** and demonstrates a complete machine learning workflow from data preprocessing and model training to evaluation and deployment using Streamlit.

---

## 📌 Project Overview

The goal of this project is to build a **binary classification model** that predicts diabetes using eight patient-related features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

The target variable is:

* `0` → No Diabetes
* `1` → Diabetes

The project uses **Logistic Regression** as the machine learning algorithm.

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

* **Rows:** 768
* **Features:** 8
* **Target:** Outcome
* **Problem Type:** Binary Classification

### Features

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skin fold thickness  |
| Insulin                  | 2-hour serum insulin         |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Patient age                  |
| Outcome                  | Diabetes prediction target   |

---

## ⚙️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Google Colab**

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Feature & Target Separation
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save Model & Scaler
   ↓
Streamlit Web Application
```

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was used because this project is a **binary classification problem**.

The model predicts the probability of a patient belonging to the diabetic class and then generates a final classification.

The data was split using:

* **80% Training Data**
* **20% Testing Data**
* `random_state = 42`
* Stratified splitting

Before training, the features were standardized using **StandardScaler**.

---

## 📈 Model Performance

The trained Logistic Regression model achieved:

### Accuracy

**71.43%**

### Classification Report

| Class           | Precision | Recall | F1-Score |
| --------------- | --------: | -----: | -------: |
| 0 — No Diabetes |      0.76 |   0.82 |     0.79 |
| 1 — Diabetes    |      0.61 |   0.52 |     0.56 |

### Confusion Matrix

```text
[[82 18]
 [26 28]]
```

This represents:

* **82** True Negatives
* **18** False Positives
* **26** False Negatives
* **28** True Positives

The model correctly classified **110 out of 154** test samples.

---

## 💾 Saved Model Files

The trained machine learning components are saved using Joblib:

```text
diabetes_model.pkl
scaler.pkl
```

### `diabetes_model.pkl`

Contains the trained Logistic Regression model.

### `scaler.pkl`

Contains the fitted StandardScaler used to standardize the input features.

---

## 🌐 Streamlit Application

A Streamlit interface was created to allow users to enter patient information and receive a prediction.

The application accepts:

```text
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
```

It then displays:

* Diabetes prediction
* Estimated probability

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/diabetes-prediction.git
```

Move into the project directory:

```bash
cd diabetes-prediction
```

---

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib streamlit
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

### 3. Train the Model

Run the training notebook/script that generates:

```text
diabetes_model.pkl
scaler.pkl
```

Make sure these files are available in the location expected by `app.py`.

---

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Project Structure

```text
diabetes-prediction/
│
├── app.py
├── diabetes_model.pkl
├── scaler.pkl
├── diabetes.csv
├── requirements.txt
├── README.md
│
└── notebook/
    └── diabetes_prediction.ipynb
```

> Adjust the folder structure if your GitHub repository uses different filenames or folders.

---

## 🖥️ Example Prediction

The user enters patient information through the Streamlit interface.

Example inputs:

```text
Pregnancies: 2
Glucose: 120
Blood Pressure: 70
Skin Thickness: 30
Insulin: 100
BMI: 32.0
Diabetes Pedigree Function: 0.5
Age: 35
```

The trained model processes these values and returns a prediction.

---

## ⚠️ Limitations

This project is intended for **educational and demonstration purposes**.

The model:

* Has an accuracy of 71.43% on the test set.
* Produces false-positive and false-negative predictions.
* Has a 52% recall for the diabetes class.
* Has not been clinically validated.
* Was trained on a relatively small dataset.
* Should not be used for medical diagnosis or treatment decisions.

---

## 🔮 Future Improvements

Possible improvements include:

* Testing additional machine learning algorithms.
* Hyperparameter tuning.
* Cross-validation.
* Improving recall for the diabetes class.
* Feature engineering.
* Class-weighting techniques.
* Model explainability using SHAP.
* Testing on larger and more diverse datasets.
* Deploying the Streamlit application online.

---

## 🎯 Learning Outcomes

Through this project, the following concepts were practiced:

* Data loading and exploration
* Feature and target selection
* Train/test splitting
* Feature scaling
* Logistic Regression
* Classification metrics
* Confusion matrix
* Model persistence using Joblib
* Streamlit application development
* Machine learning model deployment

---

## 👨‍💻 Author

**Muhammad Abdul Rehman Imran**

AI/ML Intern — **Zynvex Solutions**

---

## 📄 License

This project is intended for educational purposes.
