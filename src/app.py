import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Carsify - Estimation de prix", layout="centered"
)

st.title("🚗 Carsify — Estimation de Prix de Véhicule")
st.markdown("Entrez les caractéristiques de votre véhicule pour obtenir une estimation de sa valeur.")



@st.cache_resource
def load_model():
    return joblib.load("models/best_car_price_model.pkl")

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modele : {e}")
    st.stop()


with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
            year = st.number_input(
                "Année de mise en circulation",
                min_value=1990,
                max_value=2026,
                value=2018,
                step=1,
            )
            km_driven = st.number_input(
                "Kilométrage (km)",
                min_value=0,
                max_value=1000000,
                value=65000,
                step=1000,
            )
            fuel = st.selectbox(
                "Type de carburant", ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
            )

    with col2:
        seller_type = st.selectbox(
            "Type de vendeur",
            ["Individual", "Dealer", "Trustmark Dealer"],
        )
        transmission = st.selectbox(
            "Boîte de vitesse", ["Manual", "Automatic"]
        )
        owner = st.selectbox(
            "Nombre de propriétaires",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner",
                "Test Drive Car",
            ],
        )

        submit_button = st.form_submit_button("💡 Estimer le prix")

# Prédiction
if submit_button:
    # Structuration des données d'entrée
    input_data = pd.DataFrame(
        [
            {
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
            }
        ]
    )

    # Calcul de la prédiction
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success(f"### Prix estimé : **{prediction:,.2f} DH**")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")