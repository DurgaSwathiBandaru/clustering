import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Streamlit App Title
st.title("Hierarchical Clustering Web App")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Preview of Dataset", df.head())

    # Data Preprocessing
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric.fillna(df_numeric.mean(), inplace=True)

    # Standardizing the Data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # PCA for Dimensionality Reduction
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(df_pca, columns=["PC1", "PC2"])

    # Performing Hierarchical Clustering
    st.write("### Choose Number of Clusters")
    n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df_pca["Hierarchical_Cluster"] = clustering.fit_predict(df_pca)

    # Saving the Model
    with open("hierarchical_model.pkl", "wb") as file:
        pickle.dump((scaler, pca, clustering), file)
    st.success("Hierarchical Model Saved Successfully!")

    # Cluster Visualization
    st.write("### Clustered Data")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["Hierarchical_Cluster"], cmap="viridis")
    plt.colorbar(scatter)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Hierarchical Clustering Results")
    st.pyplot(fig)

    # Predicting New Data
    st.write("### Predict Cluster for New Data")

    feature_inputs = []
    for col in df_numeric.columns:
        min_val = float(df_numeric[col].min())
        max_val = float(df_numeric[col].max())
        mean_val = float(df_numeric[col].mean())

        # Ensure slider min < max
        if min_val == max_val:
            min_val = mean_val - 1
            max_val = mean_val + 1

        feature_inputs.append(st.slider(f"Select {col}", min_val, max_val, mean_val))

    # Transform Input Data
    new_data = np.array(feature_inputs).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data)
    new_data_pca = pca.transform(new_data_scaled)

    # Predict Cluster without refitting
    distances = np.linalg.norm(df_pca[['PC1', 'PC2']].values - new_data_pca, axis=1)
    predicted_cluster = df_pca["Hierarchical_Cluster"].iloc[np.argmin(distances)]

    st.write(f"### Predicted Cluster: {predicted_cluster}")

    # Compute Silhouette Score
    silhouette = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['Hierarchical_Cluster'])
    st.write(f"### Silhouette Score: {silhouette:.4f}")

st.write("Developed by Swathi 🚀")
