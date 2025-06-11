import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

def show_fp_growth():
    st.title("FP-Growth – Règles d'association")
st.title("\U0001F4CA Segmentation RFM des clients")

    # Téléversement du fichier
    st.markdown("### \U0001F4C1 Charger votre fichier CSV")
    fichier_csv = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if fichier_csv is not None:
        try:
            df = pd.read_csv(fichier_csv, encoding='ISO-8859-1')
            st.success(" Fichier chargé avec succès.")
        except Exception as e:
            st.error(f" Erreur de chargement : {e}")
            return
    else:
        st.warning(" Veuillez charger un fichier CSV pour continuer.")
        return
    st.write(df.head())

    st.subheader("2. Nettoyage et préparation des données")
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    st.write("Données après suppression des valeurs manquantes :")
    st.write(df.head())

    st.subheader("3. Visualisation des données")
    fig, ax = plt.subplots(figsize=(10, 6))
    df['InvoiceNo'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top 10 des factures")
    ax.set_xlabel("Numéro de facture")
    ax.set_ylabel("Nombre d'articles")
    st.pyplot(fig)

    st.subheader("4. Transformation en panier (Basket)")
    basket = df[df['Country'] == 'France'].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    st.write("Extrait du panier :")
    st.write(basket.head())

    st.subheader("5. Binarisation des données")
    basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)
    st.write(basket_sets.head())

    st.subheader("6. Algorithme FP-Growth")
    min_support = st.slider("Support minimal", 0.01, 0.2, 0.05, 0.01)
    frequent_itemsets = fpgrowth(basket_sets, min_support=min_support, use_colnames=True)
    st.write("Itemsets fréquents :")
    st.dataframe(frequent_itemsets)

    st.subheader("7. Génération des règles d'association")
    metric = st.selectbox("Métrique", ["lift", "confidence", "support"])
    min_threshold = st.slider("Seuil minimal", 0.1, 1.0, 0.6, 0.05)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if not rules.empty:
        st.write(f"Règles d'association (basées sur {metric}) :")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        st.subheader("8. Visualisation des règles")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", ax=ax2, legend=False)
        ax2.set_title("Support vs Confidence")
        st.pyplot(fig2)
    else:
        st.warning("Aucune règle trouvée avec les paramètres actuels.")

    st.success("Analyse FP-Growth terminée.")
