# 💰 Health Insurance Premium Predictor

An intelligent web application built with Streamlit that estimates individual health insurance premiums based on demographic, lifestyle, and medical inputs. By using separate regression models for different age segments, it delivers more tailored and accurate predictions.

---

## 🌐 Live Demo
Try it out here: **[Health Insurance Premium Predictor](https://rakesh-project-insurance-premium-predictor.streamlit.app)**  
🎥 Watch the full presentation: **[Project Presentation](./Health-Insurance-Presentation.pdf)**

---

## 🛠 Features

- Interactive and clean Streamlit interface.
- Predicts premium cost based on user inputs like age, BMI, smoking status, income, etc.
- Dual-model approach for better accuracy:
    - Linear Regression for younger users
    - XGBoost for the rest of the population
- Uses pre-trained and serialized models & scalers for real-time predictions
- Lightweight, fast, and easy to run locally
- No database or backend server required

---


## 🗂️ Project Structure

 ``` Car_Damage_Detector/
Health_Insurance_Cost_Predictor/
│
├── artifacts/                      # Serialized models and scalers
│   ├── model_rest.joblib           # XGBoost model for the general adult population
│   ├── model_young.joblib          # Linear Regression model for younger users
│   ├── scaler_rest.joblib          # StandardScaler fitted on “rest” training data
│   └── scaler_young.joblib         # StandardScaler fitted on “young” training data
│
├── LICENSE                         # Apache License file
├── ML_Premium_Prediction.ipynb     # Initial Notebook with extreme error 
├── README.md                       # This documentation
├── main.py                         # Streamlit app logic
├── prediction_helper.py            # Model loading and prediction logic
└── requirements.txt                # Python dependencies
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8+

### Clone the Repository
```bash
https://github.com/rakeshkapilavayi/Health-Insurance-Premium-Predictor.git
cd Health-Insurance-Premium-Predictor
```
### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🧠 How It Works

### 1. User Inputs

- **Age (years)**
- **Number of Dependents**
- **Income in Lakhs**
- **Genetical Risk**  
  *(0 = none, 1 = low, 2 = moderate, 3 = high)*
- **Insurance Plan**  
  *(e.g., Bronze, Silver, Gold)*
- **Employment Status**  
  *(e.g., Salaried, Self-Employed, Unemployed)*
- **Gender**  
  *(Male/Female)*
- **Marital Status**  
  *(Married/Unmarried)*
- **BMI Category**  
  *(Underweight, Normal, Overweight, Obese)*
- **Smoking Status**  
  *(No Smoking / Smoker)*
- **Region**  
  *(Northeast, Northwest, Southeast, Southwest)*
- **Medical History**  
  *(No Disease / Has Disease)*

### 2. Segmentation Logic

- **If Age ≤ 25**
  - Uses `scaler_young.joblib`
  - Uses `model_young.joblib` (Linear Regression)

- **If Age > 25**
  - Uses `scaler_rest.joblib`
  - Uses `model_rest.joblib` (XGBoost Regressor)


### 3. Prediction Flow

1. Input data is preprocessed using the appropriate scaler.
2. Features are passed to the corresponding model.
3. The predicted premium cost is returned instantly.

---

## Deployment
[▶️ Watch demo video](https://youtu.be/g4G2WGpS4yU)


