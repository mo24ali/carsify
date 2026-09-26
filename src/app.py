import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Carsify - Estimation de prix", layout="centered"
)

st.title("🚗 Carsify — Estimation de Prix de Véhicule")
st.markdown("Entrez les caractéristiques de votre véhicule pour obtenir une estimation de sa valeur.")
