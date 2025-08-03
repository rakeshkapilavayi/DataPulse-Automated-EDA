# DataPulse: Your Easy Data Analysis Tool

Hello! I'm **Rakesh Kapilavayi**, and I created **DataPulse** to help you explore and understand your data in a simple way. This tool is a web app that lets you:

- Upload a dataset (CSV or Excel)
- Clean it manually or automatically
- Create visualizations
- Run machine learning models
- Gain useful insights

Whether you're new to data or an expert, this app makes data analysis fun and easy!

## 🌐 Live Demo
Try it out here: **[Click Here](https://rakesh-project-insurance-premium-predictor.streamlit.app)**  

---

## 🚀 What Can DataPulse Do?

### 📂 Upload Your Data
- Add a CSV or Excel file to start.

### 📊 See a Summary
- Number of rows and columns
- Missing values
- Duplicate rows

### 🧹 Clean Your Data

#### Manual Cleaning
- Choose how to handle missing values (mean, median, drop rows, etc.)
- Delete duplicates
- Checkbox turns **light blue** when selected

#### Auto Cleaning
- Automatically:
  - Handle missing values
  - Remove duplicates
  - Deal with outliers

### 📈 Explore Your Data (EDA)
- Histograms for numerical columns
- Scatter plot for the two most correlated numeric columns
- Heatmap showing correlation
- Bar charts for categorical columns
- Box plots to detect outliers

### 🤖 Run Machine Learning

#### Choose Type
- **Classification** (predict categories)
- **Regression** (predict numbers)

#### Select a Model
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM

- Auto-handles categorical variables (for classification)
- View accuracy metrics and visual reports
- Option to tune model (for better performance; slower)
  - Checkbox turns **light blue** when selected

### 💡 Get Insights
- Understand:
  - Missing value patterns
  - Correlations
  - Data quality
- Get tips on what to do next

### 💾 Save Your Data
- Download the cleaned dataset as a CSV file

---

## 🛠 What You Need

- A computer with **Python 3.8** or higher
- A web browser (Chrome, Firefox, etc.)

---

## ⚙️ How to Set It Up

### 1. Get the Files

Download or clone the project:

```bash
https://github.com/rakeshkapilavayi/DataPulse-Automated-EDA.git
cd DataPulse-Automated-EDA
```
### 2. Set Up a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Requirements
```bash
pip install -r requirements.txt
```

This will install:

- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Openpyxl
- XGBoost

### 4. Add a Logo (optional)
Place a **logo.png** file inside the **datapulse** folder.
Don’t have a logo?
Remove or comment out the **st.sidebar.image()** line in **interface.py**.

## ▶️ How to Use DataPulse

### 1. Start the App
```bash
streamlit run interface.py
```
It will usually open at: [http://localhost:8501](http://localhost:8501)

## 📤 2. Upload a Dataset
- Click **"Choose a CSV or Excel file"**.
- You’ll see a preview of the dataset.
- Uploading a new file **resets the app**.

## 🧭 3. Use the Tabs

### 🔍 Summary
- Basic stats on rows, columns, missing data, etc.

### 🛠 Manual Cleaning
- Fix missing data or remove duplicates manually.

### ⚙️ Auto Cleaning
- Clean your dataset automatically.

### 📊 EDA (Exploratory Data Analysis)
- Visualize your data:
  - Scatter plots
  - Histograms
  - Heatmaps
  - Bar charts

### 🚨 Outliers
- Detect outliers using **box plots**.

### 🤖 Machine Learning
- Choose task: **Classification** or **Regression**
- Select **target column**
- Pick a **model** (dropdown highlights the selected one)
- Option to **tune model** (checkbox appears)
- Click **Train Model** to view results

### 💡 Insights
- Get smart tips and observations from your data.

### 💾 Export
- Download the cleaned dataset as a **CSV file**.

---

## 📁 Project Files

Here’s what’s inside the `datapulse` folder:
 ``` Car_Damage_Detector/
Health_Insurance_Cost_Predictor/
│
├── interface.py        # Main Streamlit app
├── functions.py        # Data cleaning, EDA, insights logic
├── machinelearning.py  # ML models and training code
├── requirements.txt    # Python dependencies
├── eda_app.log         # App log file (for debugging)
├── logo.png            # Optional sidebar logo
├── README.md           # You’re reading it!
└── requirements.txt    # Python dependencies
```

---

Let me know if you want a version with badges (for license, Python version, etc.), or if you'd like help with GitHub Pages or documentation layout.

