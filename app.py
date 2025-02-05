import os
import sys

# Überprüfe, ob Joblib installiert ist, wenn nicht, installiere es
try:
    import joblib
except ModuleNotFoundError:
    os.system(f"{sys.executable} -m pip install joblib")
    import joblib  # Nochmals versuchen nach der Installation

import streamlit as st
import pandas as pd
import numpy as np

# Lade das Modell
model = joblib.load("nfp_forecast_model.pkl")

# Streamlit-Web-App
st.title("📊 Makroökonomische Vorhersagen: ADP, NFP, PMI & FOMC")

st.write("🚀 KI-gestützte Prognosen für makroökonomische Indikatoren")

# Nutzer-Eingaben für Vorhersage
jobless_claims = st.slider("Erstanträge auf Arbeitslosenhilfe", min_value=100, max_value=800, value=300)
pmi = st.slider("PMI-Wert", min_value=40.0, max_value=60.0, value=52.0)
bond_yield = st.slider("US-Anleiherendite (%)", min_value=0.5, max_value=5.0, value=2.5)
fomc_rate = st.slider("FOMC-Zinssatz (%)", min_value=0.5, max_value=5.0, value=3.0)

# Daten für Vorhersage formatieren
input_data = pd.DataFrame([[jobless_claims, pmi, bond_yield, fomc_rate]],
                          columns=["Jobless_Claims", "PMI", "Bond_Yield", "FOMC_Rate"])

# Vorhersage berechnen
prediction = model.predict(input_data)[0]

st.subheader("📈 Prognose für NFP")
st.write(f"📌 Erwarteter Non-Farm Payrolls-Wert: **{prediction:.0f} Jobs**")

st.write("🔄 Daten werden kontinuierlich aktualisiert.")
