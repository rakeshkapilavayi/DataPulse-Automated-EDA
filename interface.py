import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import plotly.express as px
from functions import get_summary, manual_cleaning, auto_clean, perform_eda, generate_insights, get_download_link
from machinelearning import train_model

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="DataPulse: Automated EDA", layout="wide", initial_sidebar_state="expanded")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eda_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced styling with improved selectbox and checkbox visibility
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox div[data-baseweb="select"] > div, 
    .stFileUploader label {
        background-color: #ffffff;
        color: #1a3c6d !important;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #1a3c6d !important;
    }
    /* Style the selected option in the selectbox */
    .stSelectbox div[data-baseweb="select"] > div[aria-expanded="true"],
    .stSelectbox div[data-baseweb="select"] > div:hover {
        background-color: #e6f3ff !important;
        border-color: #007bff !important;
    }
    .stSelectbox div[data-baseweb="select"] ul[role="listbox"] li[aria-selected="true"] {
        background-color: #cce5ff !important;
        color: #1a3c6d !important;
    }
    /* Checkbox styling */
    .stCheckbox div[role="checkbox"] {
        background-color: #ffffff;
        color: #1a3c6d !important;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stCheckbox div[role="checkbox"] > div {
        color: #1a3c6d !important;
    }
    /* Ensure checked state is visible */
    .stCheckbox div[role="checkbox"][aria-checked="true"] {
        background-color: #e6f3ff !important;
        border-color: #007bff !important;
    }
    .stCheckbox div[role="checkbox"][aria-checked="true"] > div {
        background-color: #007bff !important;
        border-color: #007bff !important;
    }
    h1, h2, h3 {
        color: #1a3c6d;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTabs > div > button {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        color: #1a3c6d;
    }
    .stTabs > div > button:hover {
        background-color: #ced4da;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stSpinner > div {
        border-color: #007bff !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with branding and info
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("""
# DataPulse: Automated EDA
**Created by**: Rakesh Kapilavayi  
**About Me**:  
- **Role**: Aspiring Data Scientist  
- **Skills**: Python, SQL, Data Cleaning, EDA, Visualization (Plotly), Machine Learning (Scikit-learn), Streamlit  
- **Contact**:  
  - Email: rakeshkapilavayi978@gmail.com  
  - LinkedIn: [Rakesh Kapilavayi](https://www.linkedin.com/in/rakesh-kapilavayi-48b9a0342/)  
  - GitHub: [rakeshkapilavayi](https://github.com/rakeshkapilavayi)  

**Project Overview**:  
This app allows users to:  
- Upload CSV/Excel files for analysis  
- Perform manual and automated data cleaning  
- Conduct interactive EDA with Plotly visualizations  
- Detect outliers  
- Train machine learning models (classification/regression)  
- Generate automated insights and recommendations  
- Export cleaned datasets  
""", unsafe_allow_html=True)

# Set page title
st.title("DataPulse: Automated Exploratory Data Analysis")

# File upload with progress feedback
with st.container():
    st.markdown("### Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key="file_uploader")
    
    # Store the name of the uploaded file to detect changes
    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = None
    
    if uploaded_file is not None:
        # Check if a new file is uploaded
        if uploaded_file.name != st.session_state['last_uploaded_file']:
            # Reset session state for dataset-related variables
            st.session_state['df'] = None
            st.session_state['cleaned_df'] = None
            st.session_state['model'] = None
            st.session_state['features'] = None
            st.session_state['task_type'] = None
            st.session_state['label_encoder'] = None
            st.session_state['last_uploaded_file'] = uploaded_file.name
            logger.info(f"New dataset uploaded: {uploaded_file.name}. Session state reset.")
        
        try:
            with st.spinner("Loading dataset..."):
                # Load the dataset
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Initialize or update session state
                st.session_state['df'] = df
                st.session_state['cleaned_df'] = df.copy()
                logger.info(f"Dataset '{uploaded_file.name}' loaded successfully. Shape: {df.shape}")

                # Display dataset preview
                st.markdown("### Dataset Preview")
                st.dataframe(st.session_state['cleaned_df'].head(), use_container_width=True)

                # Create tabs for functionalities
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "Summary", "Manual Cleaning", "Auto Cleaning", "EDA", "Outliers", 
                    "Machine Learning", "Insights"
                ])

                # Tab 1: Summary
                with tab1:
                    st.markdown("### Dataset Summary")
                    try:
                        summary = get_summary(st.session_state['cleaned_df'])
                        summary_df = pd.DataFrame({
                            "Column Name": summary['columns'],
                            "Data Type": [summary['dtypes'][col] for col in summary['columns']],
                            "Missing Values": [summary['missing_values'][col] for col in summary['columns']],
                            "Unique Values": [st.session_state['cleaned_df'][col].nunique() for col in summary['columns']]
                        })
                        
                        st.markdown("#### Column Information")
                        st.dataframe(
                            summary_df.style.set_properties(**{'text-align': 'left'}),
                            use_container_width=True
                        )
                        st.markdown(f"**Total Rows**: {summary['shape'][0]}")
                        st.markdown(f"**Total Columns**: {summary['shape'][1]}")
                        st.markdown(f"**Duplicate Rows**: {summary['duplicates']}")
                        logger.info("Summary tab displayed successfully")
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        logger.error(f"Summary error: {e}")

                # Tab 2: Manual Cleaning
                with tab2:
                    st.markdown("### Manual Data Cleaning")
                    missing_actions = {}
                    missing_columns = st.session_state['cleaned_df'].columns[st.session_state['cleaned_df'].isnull().any()].tolist()
                    
                    if missing_columns:
                        st.markdown("#### Handle Missing Values")
                        for col in missing_columns:
                            with st.expander(f"Column: {col} (Missing: {st.session_state['cleaned_df'][col].isnull().sum()})"):
                                action = st.selectbox(
                                    f"Action for {col}",
                                    ["None", "Drop", "Mean", "Median", "Mode"],
                                    key=f"missing_{col}"
                                )
                                if action != "None":
                                    missing_actions[col] = action
                    else:
                        st.info("No missing values found in the dataset.")
                    
                    remove_duplicates = st.checkbox("Remove Duplicate Rows", key="remove_duplicates")
                    
                    if st.button("Apply Manual Cleaning", key="apply_manual"):
                        try:
                            with st.spinner("Applying manual cleaning..."):
                                st.session_state['cleaned_df'] = manual_cleaning(
                                    st.session_state['cleaned_df'], missing_actions, remove_duplicates
                                )
                                st.success("Manual cleaning applied successfully!")
                                st.markdown("### Cleaned Dataset Preview")
                                st.dataframe(st.session_state['cleaned_df'].head(), use_container_width=True)
                                logger.info("Manual cleaning applied successfully")
                        except Exception as e:
                            st.error(f"Error applying manual cleaning: {e}")
                            logger.error(f"Manual cleaning error: {e}")

                # Tab 3: Auto Cleaning
                with tab3:
                    st.markdown("### Automated Data Cleaning")
                    if st.button("Perform Auto Cleaning", key="auto_clean"):
                        try:
                            with st.spinner("Performing auto cleaning..."):
                                st.session_state['cleaned_df'], report = auto_clean(st.session_state['cleaned_df'])
                                st.success("Auto cleaning completed!")
                                st.markdown(f"**New Shape**: {st.session_state['cleaned_df'].shape[0]} rows, {st.session_state['cleaned_df'].shape[1]} columns")
                                st.markdown("#### Cleaning Report")
                                st.write(f"- Missing Values Handled: {len(report['missing_handled'])} columns")
                                for col, method in report['missing_handled'].items():
                                    st.write(f"  - {col}: {method}")
                                st.write(f"- Duplicates Removed: {report['duplicates_removed']}")
                                st.write(f"- Outliers Capped: {len(report['outliers_capped'])} columns")
                                st.markdown("### Cleaned Dataset Preview")
                                st.dataframe(st.session_state['cleaned_df'].head(), use_container_width=True)
                                logger.info("Auto cleaning completed successfully")
                        except Exception as e:
                            st.error(f"Error during auto cleaning: {e}")
                            logger.error(f"Auto cleaning error: {e}")

                # Tab 4: EDA
                with tab4:
                    st.markdown("### Exploratory Data Analysis")
                    try:
                        figures = perform_eda(st.session_state['cleaned_df'])
                        
                        if 'histograms' in figures:
                            st.markdown("#### Distributions")
                            for fig in figures['histograms']:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        if 'scatter' in figures:
                            st.markdown("#### Correlation Scatter Plot")
                            st.plotly_chart(figures['scatter'], use_container_width=True)
                        
                        if 'heatmap' in figures:
                            st.markdown("#### Correlation Heatmap")
                            st.plotly_chart(figures['heatmap'], use_container_width=True)
                        
                        if 'categorical' in figures:
                            st.markdown("#### Categorical Distributions")
                            for fig in figures['categorical']:
                                st.plotly_chart(fig, use_container_width=True)
                        logger.info("EDA visualizations displayed successfully")
                    except Exception as e:
                        st.error(f"Error generating EDA visualizations: {e}")
                        logger.error(f"EDA error: {e}")

                # Tab 5: Outliers
                with tab5:
                    st.markdown("### Outlier Detection")
                    num_columns = st.session_state['cleaned_df'].select_dtypes(include=['float64', 'int64']).columns
                    if num_columns.size > 0:
                        for col in num_columns:
                            fig = px.box(
                                st.session_state['cleaned_df'], 
                                y=col, 
                                title=f'Box Plot of {col}',
                                color_discrete_sequence=['#00CC96']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        logger.info("Outlier visualizations displayed successfully")
                    else:
                        st.info("No numerical columns available for outlier detection.")

                # Tab 6: Machine Learning
                with tab6:
                    st.markdown("### Machine Learning")
                    try:
                        # Initialize session state for task_type if not set
                        if 'task_type' not in st.session_state:
                            st.session_state['task_type'] = None

                        # Task type selection
                        task_type = st.selectbox(
                            "Select Task Type", 
                            ["Select a task type", "Classification", "Regression"], 
                            key="task_type_select",
                            index=0 if st.session_state['task_type'] is None else 
                                (1 if st.session_state['task_type'] == 'classification' else 2)
                        )

                        # Update session state based on selection
                        if task_type != "Select a task type":
                            st.session_state['task_type'] = task_type.lower()
                        else:
                            st.session_state['task_type'] = None

                        if st.session_state['task_type'] is None:
                            st.warning("Please select a task type (Classification or Regression).")
                        else:
                            # Filter target columns based on the cleaned dataset
                            target_columns = (
                                st.session_state['cleaned_df'].select_dtypes(include=['object', 'category']).columns.tolist()
                                if st.session_state['task_type'] == "classification"
                                else st.session_state['cleaned_df'].select_dtypes(include=['float64', 'int64']).columns.tolist()
                            )
                            model_options = (
                                ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "DecisionTreeClassifier", "SVC"]
                                if st.session_state['task_type'] == "classification"
                                else ["LinearRegression", "RandomForestRegressor", "XGBRegressor", "DecisionTreeRegressor", "SVR"]
                            )

                            if target_columns:
                                # Target column selection
                                target_column = st.selectbox(
                                    "Select Target Column", 
                                    ["Select a target column"] + target_columns, 
                                    key="target_column_select",
                                    index=0
                                )
                                # Model selection
                                model_type = st.selectbox(
                                    "Select Model", 
                                    ["Select a model"] + model_options, 
                                    key="model_type_select",
                                    index=0
                                )
                                # Hyperparameter tuning option
                                tune_params = st.checkbox("Enable Hyperparameter Tuning (Slower)", key="tune_params")
                                # Display selected options
                                if st.session_state['task_type'] and target_column != "Select a target column" and model_type != "Select a model":
                                    st.info(f"**Selected Options**: Task Type = {st.session_state['task_type'].capitalize()}, "
                                            f"Target Column = {target_column}, Model = {model_type}, "
                                            f"Tuning = {'On' if tune_params else 'Off'}")
                                
                                # Train Model button
                                if st.button("Train Model", key="train_model", 
                                            disabled=not (st.session_state['task_type'] and 
                                                        target_column != "Select a target column" and 
                                                        model_type != "Select a model")):
                                    with st.spinner("Training model..."):
                                        try:
                                            model, report, cm, cm_fig, features, label_encoder = train_model(
                                                st.session_state['cleaned_df'], 
                                                target_column, 
                                                st.session_state['task_type'], 
                                                model_type,
                                                tune_params=tune_params
                                            )
                                            st.session_state['model'] = model
                                            st.session_state['features'] = features
                                            st.session_state['task_type'] = st.session_state['task_type']
                                            st.session_state['label_encoder'] = label_encoder
                                            st.success("Model trained successfully!")
                                            
                                            st.markdown("#### Model Evaluation")
                                            if st.session_state['task_type'] == 'classification':
                                                st.write("**Classification Report**:")
                                                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                                                if cm is not None:
                                                    st.markdown("#### Confusion Matrix")
                                                    st.plotly_chart(cm_fig, use_container_width=True)
                                            else:
                                                st.write("**Regression Metrics**:")
                                                st.write(f"- Mean Squared Error: {report['Mean Squared Error']:.4f}")
                                                st.write(f"- Mean Absolute Error: {report['Mean Absolute Error']:.4f}")
                                                st.write(f"- R² Score: {report['R² Score']:.4f}")
                                                st.write(f"- Cross-Validation Score: {report['Cross_Validation_Score']:.4f}")
                                            if 'Feature_Importance' in report:
                                                st.markdown("#### Feature Importance")
                                                st.dataframe(pd.DataFrame(report['Feature_Importance']), use_container_width=True)
                                            logger.info(f"Model {model_type} trained for {st.session_state['task_type']}")
                                        except Exception as e:
                                            st.error(f"Error training model: {e}")
                                            logger.error(f"Model training error: {e}")
                            else:
                                st.warning(f"No suitable columns for {st.session_state['task_type'].capitalize()}. Please check your dataset.")
                    except Exception as e:
                        st.error(f"Machine learning error: {e}")
                        logger.error(f"Machine learning error: {e}")

                # Tab 7: Insights
                with tab7:
                    st.markdown("### Data Insights & Recommendations")
                    try:
                        insights, recommendations = generate_insights(st.session_state['cleaned_df'])
                        st.markdown("#### Key Insights")
                        if insights:
                            for insight in insights:
                                st.markdown(f"- {insight}")
                        else:
                            st.info("No significant insights generated.")
                        
                        st.markdown("#### Recommendations")
                        if recommendations:
                            for recommendation in recommendations:
                                st.markdown(f"- {recommendation}")
                        else:
                            st.info("No recommendations generated.")
                        logger.info("Insights and recommendations displayed successfully")
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")
                        logger.error(f"Insights error: {e}")

                # Download Cleaned Dataset
                st.markdown("### Export Cleaned Dataset")
                try:
                    st.markdown(
                        get_download_link(st.session_state['cleaned_df'], filename=f"cleaned_{uploaded_file.name}"),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error generating download link: {e}")
                    logger.error(f"Download link error: {e}")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            logger.error(f"Dataset loading error: {e}")
    else:
        # Clear session state when no file is uploaded
        if 'last_uploaded_file' in st.session_state:
            st.session_state['last_uploaded_file'] = None
            st.session_state['df'] = None
            st.session_state['cleaned_df'] = None
            st.session_state['model'] = None
            st.session_state['features'] = None
            st.session_state['task_type'] = None
            st.session_state['label_encoder'] = None
            logger.info("Session state cleared due to no file uploaded.")
        st.info("Please upload a CSV or Excel file to start analyzing.")