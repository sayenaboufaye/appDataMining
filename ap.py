# ap.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from KMeans import show_kmeans
from RFM import show_rfm

st.title("Projet Data Mining")

menu = ["Accueil", "K-means", "RFM", "FP-Growth"]
model_choice = st.sidebar.selectbox("Choisissez une méthode", menu)

@st.cache_data
def load_data():
    df = pd.read_csv("donnees_ecommerce.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

def show_home():
    st.subheader("Bienvenue dans le projet de Data Mining")
    st.write("""
        Cette application Streamlit vous permet d'explorer différentes techniques de data mining :
        - K-means clustering
        - Segmentation RFM
        - Règles d'association (FP-Growth)
    """)

def show_fp_growth():
    st.title("FP-Growth – Règles d'association")
    df = load_data()
    df = df[df['Quantity'] > 0]

    st.subheader("Transformation en panier")
    basket = df[df['Country'] == 'France'].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    st.write(basket.head())

    st.subheader("Binarisation")
    basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)
    st.write(basket_sets.head())

    st.subheader("Algorithme FP-Growth")
    min_support = st.slider("Support minimal", 0.01, 0.2, 0.05, 0.01)
    frequent_itemsets = fpgrowth(basket_sets, min_support=min_support, use_colnames=True)
    st.write("Itemsets fréquents :")
    st.dataframe(frequent_itemsets)

    st.subheader("Règles d'association")
    metric = st.selectbox("Métrique", ["lift", "confidence", "support"])
    min_threshold = st.slider("Seuil minimal", 0.1, 1.0, 0.6, 0.05)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if not rules.empty:
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        st.subheader("Visualisation des règles")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", ax=ax2, legend=False)
        ax2.set_title("Support vs Confidence")
        st.pyplot(fig2)
    else:
        st.warning("Aucune règle trouvée avec les paramètres actuels.")

    st.success("Analyse FP-Growth terminée.")

# Affichage selon le choix utilisateur
if model_choice == "Accueil":
    show_home()
elif model_choice == "K-means":
    show_kmeans()
elif model_choice == "RFM":
    show_rfm()
elif model_choice == "FP-Growth":
    show_fp_growth()
