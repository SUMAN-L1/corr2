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

# Function to attempt reading a CSV file with different encodings
def read_csv_with_encodings(file, encodings=['utf-8', 'latin1', 'ISO-8859-1']):
    for encoding in encodings:
        try:
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the file with the provided encodings.")

# Function to read data from uploaded file
def read_file(file, file_type):
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
            st.write(f"Correlation Matrix ({method}):")
            st.write(corr_matrix)

            # Plot Correlation Heatmap
            st.subheader(f"Correlation Heatmap ({method})")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'Correlation Heatmap ({method})')
            st.pyplot(plt.gcf())
            plt.clf()

        # Plot interactive heatmap using Plotly
        st.subheader("Interactive Correlation Heatmap (Pearson)")
        fig = px.imshow(corr_methods['Pearson'], text_auto=True, color_continuous_scale='coolwarm')
        fig.update_layout(title='Interactive Correlation Heatmap (Pearson)')
        st.plotly_chart(fig)

        # Plot pairwise scatter plots with regression lines
        st.subheader("Pairwise Scatter Plots with Regression Lines")
        pair_plot = sns.pairplot(data, kind='reg')
        st.pyplot(pair_plot.figure)
        plt.clf()

        # Plot correlation matrix with histograms
        st.subheader("Correlation Matrix with Histograms")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
        # Diagonal histograms
        for i, col in enumerate(data.columns):
            plt.subplot(len(data.columns), len(data.columns), i*len(data.columns) + i + 1)
            plt.hist(data[col].dropna(), bins=20, color='lightblue', edgecolor='black')
            plt.title(col, fontsize=10)
        st.pyplot(plt.gcf())
        plt.clf()

        # Plot Correlation Circle Plot
        st.subheader("Correlation Circle Plot")
        corr_matrix = corr_methods['Pearson']
        correlation_circle(corr_matrix)

        # Plot Clustered Heatmap
        st.subheader("Clustered Correlation Heatmap")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        corr_matrix = pd.DataFrame(scaled_data).corr()
        sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', figsize=(10, 10))
        st.pyplot(plt.gcf())
        plt.clf()

      # Plot correlation matrix with significance
st.subheader("Correlation Heatmap with Significance")
p_values = correlation_significance(data)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_methods['Pearson'], annot=True, cmap='coolwarm', fmt=".2f", mask=p_values > 0.05)
plt.title('Correlation Heatmap with Significance Mask')
st.pyplot(plt.gcf())
plt.clf()

# Interpretations
st.subheader("Interpretations")
st.write("### Correlation Matrices")
st.write("The correlation matrices show the pairwise correlation coefficients between variables using different methods (Pearson, Spearman, Kendall). Each method has its own characteristics:")
st.write(" - **Pearson**: Measures linear relationships.")
st.write(" - **Spearman**: Measures monotonic relationships, less sensitive to outliers.")
st.write(" - **Kendall**: Measures ordinal relationships, robust against ties.")

st.write("### Correlation Heatmap")
st.write("The heatmap visualizes the correlation matrices. Darker colors represent stronger correlations (positive or negative).")

st.write("### Interactive Correlation Heatmap")
st.write("Interactive heatmap allows for better exploration of correlations. Hover over cells to see values.")

st.write("### Pairwise Scatter Plots with Regression Lines")
st.write("Scatter plots with regression lines show relationships between variables. Trends or patterns in these plots can indicate correlations.")

st.write("### Correlation Matrix with Histograms")
st.write("Heatmap with histograms on the diagonal provides insights into the distribution of individual variables and their correlations.")

st.write("### Correlation Circle Plot")
st.write("The correlation circle plot visualizes the correlations between variables in a 2D space, using principal component analysis.")

st.write("### Clustered Correlation Heatmap")
st.write("Clustered heatmap groups similar variables together, making it easier to identify patterns and relationships.")

st.write("### Correlation Heatmap with Significance")
st.write("This heatmap includes a mask to highlight significant correlations at a 5% significance level. Correlations with p-values greater than 0.05 are masked.")
