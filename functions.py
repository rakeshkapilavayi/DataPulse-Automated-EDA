import base64
import pandas as pd
import numpy as np
import plotly.express as px
import logging
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

def get_summary(df):
    """Generate a summary of the dataset including column info and statistics."""
    summary = {
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'shape': df.shape,
        'duplicates': df.duplicated().sum()
    }
    return summary

def manual_cleaning(df, missing_actions, remove_duplicates):
    """Apply manual cleaning based on user-specified actions."""
    cleaned_df = df.copy()
    for col, action in missing_actions.items():
        if action == "Drop":
            cleaned_df = cleaned_df.dropna(subset=[col])
        elif action == "Mean":
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif action == "Median":
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif action == "Mode":
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    return cleaned_df

def auto_clean(df):
    """Perform automated cleaning including missing value handling, duplicate removal, and outlier capping."""
    cleaned_df = df.copy()
    report = {'missing_handled': {}, 'duplicates_removed': 0, 'outliers_capped': {}}

    # Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            if cleaned_df[col].dtype in ['float64', 'int64']:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                report['missing_handled'][col] = 'Mean'
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                report['missing_handled'][col] = 'Mode'

    # Remove duplicates
    duplicates = cleaned_df.duplicated().sum()
    if duplicates > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        report['duplicates_removed'] = duplicates

    # Cap outliers using IQR
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)].any().any():
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
            report['outliers_capped'][col] = f"Capped at {lower_bound:.2f} to {upper_bound:.2f}"

    return cleaned_df, report

def perform_eda(df):
    figures = {}
    # Numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Histograms
    figures['histograms'] = [
        px.histogram(df, x=col, title=f'Distribution of {col}', color_discrete_sequence=['#00CC96'])
        for col in num_cols
    ]
    
    # Correlation Scatter Plot
    if len(num_cols) >= 2:
        try:
            # Calculate correlation matrix
            corr_matrix = df[num_cols].corr()
            corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            max_corr = corr_matrix.stack().idxmax()
            col1, col2 = max_corr
            
            # Dynamic sampling based on dataset size
            max_rows = 5000  # Adjust this threshold as needed
            if df.shape[0] > max_rows:
                sample_df = df[[col1, col2]].sample(n=max_rows, random_state=42)
                logger.info(f"Sampled {max_rows} rows for scatter plot due to large dataset size: {df.shape[0]}")
            else:
                sample_df = df[[col1, col2]]
            
            # Dynamic plot dimensions based on dataset size
            n_rows = df.shape[0]
            n_cols = df.shape[1]
            base_height = 400
            base_width = 600
            height = min(base_height + int(n_cols * 10), 600)  # Scale height with columns, cap at 600
            width = min(base_width + int(n_cols * 20), 800)    # Scale width with columns, cap at 800
            
            # Optimize marker size and hover for large datasets
            marker_size = 8 if n_rows < 1000 else 5  # Smaller markers for larger datasets
            hover_mode = False if n_rows > 10000 else True  # Disable hover for very large datasets
            
            # Create scatter plot
            figures['scatter'] = px.scatter(
                sample_df,
                x=col1,
                y=col2,
                title=f'Scatter Plot: {col1} vs {col2} (Correlation: {corr_matrix.loc[col1, col2]:.2f})',
                color_discrete_sequence=['#EF553B'],
                height=height,
                width=width,
                render_mode='webgl',  # Use WebGL for faster rendering
                opacity=0.6,          # Slight transparency for overlapping points
                size_max=marker_size
            )
            
            # Update layout for better performance and readability
            figures['scatter'].update_traces(
                marker=dict(size=marker_size),
                hoverinfo='skip' if not hover_mode else 'all'
            )
            figures['scatter'].update_layout(
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            logger.info(f"Scatter plot generated for {col1} vs {col2} with dimensions {width}x{height}")
        except Exception as e:
            logger.error(f"Error generating scatter plot: {e}")
    
    # Correlation Heatmap
    if len(num_cols) >= 2:
        figures['heatmap'] = px.imshow(
            corr_matrix, 
            title='Correlation Heatmap', 
            color_continuous_scale='RdBu_r',
            height=500,  # Fixed height for heatmap
            width=600    # Fixed width for heatmap
        )
    
    # Categorical Distributions
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    figures['categorical'] = [
        px.histogram(df, x=col, title=f'Categorical Distribution of {col}', color_discrete_sequence=['#636EFA'])
        for col in cat_cols
    ]
    
    return figures

def generate_insights(df):
    """Generate detailed insights and recommendations based on the dataset."""
    insights = []
    recommendations = []

    # Statistical Summary
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_cols.empty:
        stats_summary = df[numeric_cols].describe()
        for col in numeric_cols:
            mean = stats_summary.loc['mean', col]
            std = stats_summary.loc['std', col]
            insights.append(f"The average value of {col} is {mean:.2f} with a standard deviation of {std:.2f}.")
            if std > mean * 0.5:
                insights.append(f"{col} shows high variability (std > 50% of mean), indicating potential outliers or diverse data points.")
            if df[col].isnull().sum() > 0:
                insights.append(f"{col} contains missing values, which may affect analysis accuracy.")

    # Correlation Analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                    insights.append(f"Strong correlation ({corr_matrix.iloc[i, j]:.2f}) between {col1} and {col2}.")
                    recommendations.append(f"Consider investigating the relationship between {col1} and {col2} for potential multicollinearity.")

    # Categorical Insights
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count < 10:
            top_value = df[col].mode()[0]
            top_freq = df[col].value_counts().iloc[0]
            insights.append(f"{col} has {unique_count} unique categories, with '{top_value}' being the most frequent ({top_freq} times, {top_freq/len(df)*100:.1f}%).")
            recommendations.append(f"Explore why '{top_value}' dominates {col} and consider its impact on downstream analysis.")

    # Outlier Detection
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            insights.append(f"{col} has {outliers} potential outliers (Z-score > 3).")
            recommendations.append(f"Review and handle outliers in {col} to improve model performance.")

    # Data Quality
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        insights.append(f"The dataset contains {missing_total} missing values across {len(df.columns[df.isnull().any()])} columns.")
        recommendations.append("Impute or drop missing values to ensure data completeness.")

    # Default messages if no specific insights
    if not insights:
        insights.append("The dataset appears to be well-balanced with no immediate anomalies.")
    if not recommendations:
        recommendations.append("No specific recommendations at this time; consider further EDA or feature engineering.")

    return insights, recommendations

def get_download_link(df, filename):
    """Generate a download link for the cleaned dataset."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href