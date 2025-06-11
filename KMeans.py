# k_means.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt

def show_kmeans():
    st.title("📊 K-means Clustering")
    st.markdown("### 📁 Charger le fichier issu de la segmentation RFM")

    fichier_csv = st.file_uploader("Fichier CSV RFM", type="csv")

    if fichier_csv is None:
        st.warning("⚠️ Veuillez charger le fichier RFM pour continuer.")
        return

    try:
        df = pd.read_csv(fichier_csv, encoding='ISO-8859-1')
        st.success("✅ Fichier chargé avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        return

    # Afficher les colonnes du fichier pour vérification
    st.write("Colonnes détectées :", df.columns.tolist())

    # On vérifie si le fichier est déjà agrégé (RFM) ou brut
    if all(col in df.columns for col in ['Recence', 'Frequence', 'Montant']):
        rfm = df.copy()
    elif all(col in df.columns for col in ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']):
        df.dropna(subset=['CustomerID'], inplace=True)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recence', 'Frequence', 'Montant']
    else:
        st.error("❌ Le fichier ne contient ni les colonnes RFM ('Recence', 'Frequence', 'Montant') "
                 "ni les colonnes brutes nécessaires ('InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice').")
        return

    # Standardisation
    scaler = StandardScaler()
    df_std = scaler.fit_transform(rfm[['Recence', 'Frequence', 'Montant']])

    # 1️⃣ Méthode du coude
    st.subheader("1️⃣ Méthode du coude pour choisir k")
    inertie = []
    range_n = range(1, 10)
    for k in range_n:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_std)
        inertie.append(kmeans.inertia_)

    fig_elbow = plt.figure(figsize=(8, 5))
    plt.plot(range_n, inertie, 'bo-')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude')
    plt.grid(True)
    st.pyplot(fig_elbow)

    # 2️⃣ Clustering K-means
    st.subheader("2️⃣ Clustering avec K-means")
    k = st.sidebar.slider("Nombre de clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(df_std)
    rfm['Cluster'] = clusters

    st.write("Aperçu des données RFM avec cluster :")
    st.dataframe(rfm.head(10), use_container_width=True)

    # 3️⃣ Visualisation 2D
    fig_2d = px.scatter(
        rfm, x='Frequence', y='Montant',
        color=rfm['Cluster'].astype(str),
        title="K-means Clusters 2D (Frequence vs Montant)",
        labels={'color': 'Cluster'}
    )
    st.plotly_chart(fig_2d)

    # 4️⃣ Répartition par cluster
    st.subheader("3️⃣ Répartition des clients par cluster")
    cluster_counts = rfm['Cluster'].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140)
    ax_pie.set_title("Distribution des clusters clients")
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    # 5️⃣ Visualisation 3D
    st.subheader("4️⃣ Visualisation 3D des clusters")
    fig_3d = px.scatter_3d(
        rfm,
        x='Recence',
        y='Frequence',
        z='Montant',
        color=rfm['Cluster'].astype(str),
        title="Clusters clients en 3D (RFM)",
        opacity=0.7
    )
    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.update_layout(legend_title_text='Cluster', margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d)
