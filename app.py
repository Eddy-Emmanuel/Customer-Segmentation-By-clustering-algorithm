import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")



df = pd.read_csv("CC GENERAL.csv")
df_id = df.pop("CUST_ID")

knn_imputer = KNNImputer()

df_filledna = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

st.markdown("<h1><center>Customer Segmentation By clustering Algorithms</center></h1>", unsafe_allow_html=True)
st.image("image.png", use_column_width=True)

selected_features = st.sidebar.multiselect(label="Select Feature for clustering", options=["None"] + df.columns.tolist())

st.markdown("# Cluster data")

nrow = st.sidebar.slider(label="Select number of rows to display", min_value=1, max_value=df_filledna.shape[0], value=5)

st.write(df_filledna[selected_features].head(nrow).style.background_gradient(cmap="viridis"))

algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["None", "KMeans", "BisectingKMeans", "MiniBatchKMeans"])

num_of_clusters = st.sidebar.slider("Number of Cluster", min_value=2, max_value=10, value=5)

scaled_data = pd.DataFrame(StandardScaler().fit_transform(df_filledna), columns=df_filledna.columns)

st.sidebar.markdown("#### Plot Settings")
selected_width = st.sidebar.slider(label="Width", min_value=5, max_value=25, value=12)
selected_height = st.sidebar.slider(label="Height", min_value=5, max_value=15, value=7)
color_palette = st.sidebar.selectbox("Select Palatte", ["viridis", "cubehelix", "magma", "plasma", "inferno"])
marker_size = st.sidebar.slider("Centriod Marker Size", min_value=50, max_value=300, value=100)

def Get_Silhouette_Score(estimator, X, n_clusters):
    algo = estimator(n_clusters=n_clusters)
    algo.fit(X) 
    return silhouette_score(X, algo.predict(X))


def Plot_Decision_Boundary(estimator, X, n_clusters, width, height, palette, ms):
    S_C = Get_Silhouette_Score(estimator, X, n_clusters)
    algo = estimator(n_clusters=n_clusters)
    reduced_data = PCA(n_components=2).fit_transform(X)
    algo.fit(reduced_data)
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    clusters = algo.predict(np.c_[xx.flatten(), yy.flatten()])
    Z = clusters.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    ax.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], c=algo.labels_, cmap=palette, s=200, edgecolors="black")
    ax.scatter(x=algo.cluster_centers_[:, 0], y=algo.cluster_centers_[:, 1], marker="x", s=ms, lw=2, color="black", cmap=palette)
    plt.title(f"Silhouette Score={S_C}")
    return st.pyplot(fig=fig)

def main():
    if len(selected_features) >= 2:
        if algorithm != "None":
            if algorithm == "KMeans":
                selected_algorithm = KMeans
            elif algorithm == "BisectingKMeans":
                selected_algorithm = BisectingKMeans
            else:
                selected_algorithm = MiniBatchKMeans

            st.markdown("# Decision Boundary")
            Plot_Decision_Boundary(selected_algorithm, 
                                scaled_data, 
                                num_of_clusters, 
                                selected_width, 
                                selected_height,
                                color_palette,
                                marker_size)

if __name__ == "__main__":
    main()