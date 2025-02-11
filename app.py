import streamlit as st
import requests
import numpy as np
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from loguru import logger
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
API_URL = "https://webappfastapi-faazh4eya0gnamew.westeurope-01.azurewebsites.net/predict"
API_KEY = os.getenv("API_KEY", "default_token")  # Récupération du token sécurisé

# Configurer les logs
logger.add("logs/streamlit_app.log", rotation="1 day", level="INFO")

# Interface Streamlit
st.title("Reconnaissance de chiffres manuscrits")

menu = st.sidebar.selectbox("Menu", ["Dessin", "Image aléatoire"])

def predict_image(image):
    """ Envoie l'image à l'API FastAPI et retourne la prédiction """
    image = np.array(image).astype("float32").flatten().tolist()  # Conversion en liste

    headers = {"x-token": API_KEY}  # Ajout du token d'authentification

    try:
        response = requests.post(API_URL, json={"data": image}, headers=headers)
        logger.info(f"📡 Requête envoyée : {len(image)} valeurs, token utilisé.")

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            logger.info(f"✅ Réponse API : {prediction}")
            return prediction
        elif response.status_code == 401:
            logger.warning("❌ Erreur 401 : Authentification échouée")
            st.error("❌ Erreur d'authentification : Token invalide ou manquant.")
        elif response.status_code == 400:
            logger.warning("⚠️ Erreur 400 : Données invalides")
            st.warning("⚠️ Données invalides : Veuillez dessiner un chiffre correct.")
        else:
            logger.error(f"🚨 Erreur inattendue : {response.json()}")
            st.error(f"🚨 Erreur inattendue : {response.json()}")

    except requests.exceptions.RequestException as e:
        logger.error(f"🌐 Erreur de connexion : {e}")
        st.error(f"🌐 Erreur de connexion à l'API : {e}")

    return None  # Retourne None en cas d'erreur

# Mode "Dessin" - Dessiner un chiffre à prédire
if menu == "Dessin":
    st.header("🎨 Dessinez un chiffre")

    canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("🔍 Prédire"):
        if canvas.image_data is not None:
            img = Image.fromarray((canvas.image_data[:, :, 0]).astype("uint8"))
            img = img.resize((28, 28))
            prediction = predict_image(img)

            if prediction is not None:
                st.write(f"📊 **Prédiction : {prediction}**")

# Mode "Image aléatoire" - Tester avec une image générée
elif menu == "Image aléatoire":
    st.header("🖼️ Prédiction sur une image aléatoire")

    if st.button("🎲 Charger une image aléatoire"):
        index = np.random.randint(0, 1000)
        image = np.random.rand(28, 28) * 255  # Image aléatoire (à remplacer par des vraies images)
        st.image(image, width=150, caption="🖼️ Image générée")

        prediction = predict_image(image)
        if prediction is not None:
            st.write(f"📊 **Prédiction : {prediction}**")
