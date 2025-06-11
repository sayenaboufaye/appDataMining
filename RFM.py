import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as gobj
import datetime as dt

def show_rfm():
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

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['CustomerID'], inplace=True)
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    donnees_ecommerce = df.query("Country=='United Kingdom'").reset_index(drop=True)
    donnees_ecommerce = donnees_ecommerce[(donnees_ecommerce['Quantity'] > 0)]

    st.subheader("1. Ventes par pays")
    ventes_par_pays = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ventes_par_pays.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('Ventes par pays')
    ax.set_xlabel('Pays')
    ax.set_ylabel('Ventes totales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("2. Meilleures ventes par mois")
    df['Month'] = df['InvoiceDate'].dt.month
    ventes_par_mois = df.groupby('Month')['TotalAmount'].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ventes_par_mois.plot(kind='bar', color='orange', edgecolor='black', ax=ax)
    ax.set_title('Meilleures ventes par mois')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Ventes totales')
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.subheader("3. Distribution des StockCode par pays")
    stock_par_pays = df.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    stock_par_pays.plot(kind='bar', color='teal', edgecolor='black', ax=ax)
    ax.set_title('Distribution des quantités de StockCode par pays')
    ax.set_xlabel('Pays')
    ax.set_ylabel('Quantité totale')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("4. Tendance des ventes (2010–2011)")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    ventes_par_mois_annee = df.groupby('YearMonth')['TotalAmount'].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    ventes_par_mois_annee.plot(marker='o', linestyle='-', color='orange', ax=ax)
    ax.set_title('Tendance générale des ventes (2010–2011)')
    ax.set_xlabel('Mois / Année')
    ax.set_ylabel('Ventes totales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("5. Top 20 articles les plus vendus (quantité)")
    top_items = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_items.plot(kind='bar', ax=ax)
    ax.set_title('Top 20 des articles les plus vendus (Quantité)')
    ax.set_xlabel('Description')
    ax.set_ylabel('Quantité totale')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("6. Articles achetés le plus souvent")
    top_transactions = df.groupby('Description')['InvoiceNo'].nunique().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_transactions.plot(kind='bar', ax=ax)
    ax.set_title('Articles achetés le plus souvent')
    ax.set_xlabel('Description')
    ax.set_ylabel('Nombre de transactions')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("7. Produit le plus vendu par pays")
    produit_max = df.groupby(['Country', 'Description'])['TotalAmount'].sum().reset_index()
    produit_max = produit_max.loc[produit_max.groupby('Country')['TotalAmount'].idxmax()]
    total_ventes_pays = df.groupby('Country')['TotalAmount'].sum()
    produit_max = produit_max.merge(total_ventes_pays, on='Country', suffixes=('', '_Total'))
    produit_max['%_Contribution'] = (produit_max['TotalAmount'] / produit_max['TotalAmount_Total']) * 100
    st.dataframe(produit_max[['Country', 'Description', 'TotalAmount', 'TotalAmount_Total', '%_Contribution']])

    st.subheader("8. Heatmap des transactions par heure et par mois")
    df['Hour'] = df['InvoiceDate'].dt.hour
    heatmap_data = df.groupby(['Month', 'Hour'])['InvoiceNo'].count().unstack()
    fig_heatmap, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, ax=ax)
    ax.set_title("Volume de transactions par mois et heure")
    st.pyplot(fig_heatmap)

    st.subheader("\U0001F3AF Analyse RFM (Récence, Fréquence, Montant)")
    date_recente = dt.datetime(2011, 12, 10)

    rfm = donnees_ecommerce.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (date_recente - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recence', 'Frequence', 'Montant']

    quartiles = rfm[['Recence', 'Frequence', 'Montant']].quantile([0.25, 0.5, 0.75]).to_dict()

    def r_score(x):
        if x <= quartiles['Recence'][0.25]: return 4
        elif x <= quartiles['Recence'][0.5]: return 3
        elif x <= quartiles['Recence'][0.75]: return 2
        else: return 1

    def fm_score(x, col):
        if x <= quartiles[col][0.25]: return 1
        elif x <= quartiles[col][0.5]: return 2
        elif x <= quartiles[col][0.75]: return 3
        else: return 4

    rfm['R'] = rfm['Recence'].apply(r_score)
    rfm['F'] = rfm['Frequence'].apply(lambda x: fm_score(x, 'Frequence'))
    rfm['M'] = rfm['Montant'].apply(lambda x: fm_score(x, 'Montant'))
    rfm['RFM_concat_score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

    def segmenter(code):
        if code == '111': return 'Clients en hibernation'
        elif code[0] == '1' and code[1] in ['1', '2']: return 'Clients à risque'
        elif code[0] == '2' and code[1] in ['3', '4']: return 'Clients à ne pas perdre'
        elif code[0] == '3' and code[1] in ['2', '3']: return "Clients presqu'endormis"
        elif code[0] == '3' and code[1] == '4': return 'Clients à suivre'
        elif code[0] == '4' and code[1] in ['3', '4']: return 'Clients loyaux'
        elif code[0] == '3' and code[1] == '1': return 'Clients prometteurs'
        elif code.startswith('41'): return 'Nouveaux clients'
        elif code.startswith('44'): return 'Clients potentiellement loyaux'
        elif code == '444': return 'Très bons clients'
        else: return 'Autres'

    rfm['Segment'] = rfm['RFM_concat_score'].apply(segmenter)

    couleurs = {
        'Clients en hibernation': 'gray',
        'Clients à risque': 'red',
        'Clients à ne pas perdre': 'brown',
        "Clients presqu'endormis": 'purple',
        'Clients à suivre': 'blue',
        'Clients loyaux': 'green',
        'Clients prometteurs': 'orange',
        'Nouveaux clients': 'cyan',
        'Clients potentiellement loyaux': 'gold',
        'Très bons clients': 'darkgreen',
        'Autres': 'pink'
    }

    graph = rfm.query("Montant < 50000 and Frequence < 2000")

    st.subheader("\U0001F4CC Visualisation des segments RFM")

    def plot_scatter(x, y, title, xlabel, ylabel):
        donnees_graphique = []
        for segment, couleur in couleurs.items():
            subset = graph[graph['Segment'] == segment]
            if not subset.empty:
                donnees_graphique.append(
                    gobj.Scatter(
                        x=subset[x],
                        y=subset[y],
                        mode='markers',
                        name=segment,
                        marker=dict(size=9, line=dict(width=1), color=couleur, opacity=0.7)
                    )
                )
        layout = gobj.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel), width=1000, height=700)
        fig = gobj.Figure(data=donnees_graphique, layout=layout)
        st.plotly_chart(fig)

    plot_scatter('Recence', 'Frequence', 'RFM : Récence vs Fréquence', 'Récence', 'Fréquence')
    plot_scatter('Recence', 'Montant', 'RFM : Récence vs Montant', 'Récence', 'Montant')
    plot_scatter('Frequence', 'Montant', 'RFM : Fréquence vs Montant', 'Fréquence', 'Montant')

    st.subheader("\U0001F4C8 Répartition des segments RFM")
    segment_counts = rfm['Segment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')
    st.pyplot(fig)

    chemin = "rfm.csv"
    rfm.to_csv(chemin, index=False, encoding='utf-8')
    st.success(" Fichier rfm.csv enregistré avec succès.")
