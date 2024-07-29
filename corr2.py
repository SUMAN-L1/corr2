import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.figure_factory as ff

# Function to calculate significance of correlation
def correlation_significance(data):
    columns = data.columns
    num_vars = len(columns)
    p_values = pd.DataFrame(np.ones((num_vars, num_vars)), columns=columns, index=columns)
    
    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                _, p_value = pearsonr(data[columns[i]], data[columns[j]])
                p_values.iloc[i, j] = p_value
    
    return p_values

# Function to calculate different types of correlation
def calculate_correlations(data):
    corr_methods = {
        'Pearson': data.corr(method='pearson'),
        'Spearman': data.corr(method='spearman'),
        'Kendall': data.corr(method='kendall')
    }
    return corr_methods

# Function to preprocess data
def preprocess_data(data):
    # Handle missing values by filling with mean
    data = data.fillna(data.mean())
    # Remove outliers using Z-score
    from scipy.stats import zscore
    z_scores = np.abs(zscore(data.select_dtypes(include=[np.number])))
    data = data[(z_scores < 3).all(axis=1)]
    return data

# Function to read data from uploaded file
def read_file(file, file_type):
    if file_type in ['csv', 'xlsx', 'xls']:
        if file_type == 'csv':
            return read_csv_with_encodings(file)
        elif file_type in ['xlsx', 'xls']:
            try:
                return pd.read_excel(file)
            except Exception as e:
                raise ValueError(f"Unable to read the Excel file. Error: {e}")
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

# Function to generate a correlation circle plot
def correlation_circle(corr_matrix):
    plt.figure(figsize=(10, 8))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(corr_matrix)
    
    # Create a dataframe for plotting
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=corr_matrix.columns)
    
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    for i, txt in enumerate(pca_df.index):
        plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]))
    plt.title('Correlation Circle Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# Streamlit app
st.title('Advanced Correlation Analysis Tool_Suman_Econ')

# Upload file
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    file_type = uploaded_file.type.split('/')[1] if '/' in uploaded_file.type else uploaded_file.type

    try:
        # Read the file
        data = read_file(uploaded_file, file_type)
        if data.empty:
            raise ValueError("The file is empty or could not be parsed.")
        
        # Display the first few rows to understand the structure
        st.write("Data preview:")
        st.write(data.head())

        # Use the first row as headers if not already set
        if data.columns[0] == 'Unnamed: 0' or data.columns[0].startswith('Unnamed'):
            data.columns = data.iloc[0]  # Use the first row as column headers
            data = data[1:]  # Remove the header row from the data

        # Reset the index
        data.reset_index(drop=True, inplace=True)

        # Check for empty data
        if data.empty:
            st.error("The data does not contain any columns after header adjustment.")
            st.stop()

        # Preprocess data
        data = preprocess_data(data)

        # Convert columns to numeric, if possible
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Compute correlation matrices
        corr_methods = calculate_correlations(data)
        
        # Display correlation matrices
        for method, corr_matrix in corr_methods.items():
           
