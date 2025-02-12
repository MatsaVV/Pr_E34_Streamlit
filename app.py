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
API_URL = "https://webappfastapi-faazh4eya0gnamew.westeurope-01.azurewebsites.net"
API_KEY = os.getenv("API_KEY", "default_token")  # Token sécurisé

# Configurer les logs
logger.add("logs/streamlit_app.log", rotation="1 day", level="INFO")

# Initialiser Streamlit
st.title("📝 Reconnaissance de chiffres manuscrits")

menu = st.sidebar.selectbox("📌 Menu", ["Dessin", "Image aléatoire", "📊 Statistiques"])

# Ajouter une variable de session pour stocker la prédiction et l'image
temp_state = st.session_state
if "prediction" not in temp_state:
    temp_state.prediction = None
if "image_data" not in temp_state:
    temp_state.image_data = None

def predict_image(image):
    """ Envoie l'image à l'API FastAPI et retourne la prédiction """
    image = np.array(image).astype("float32").flatten().tolist()
    headers = {"x-token": API_KEY}
    try:
        response = requests.post(f"{API_URL}/predict", json={"data": image}, headers=headers)
        logger.info(f"📡 Requête envoyée : {len(image)} valeurs.")

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            logger.info(f"✅ Réponse API : {prediction}")
            return prediction
        elif response.status_code == 401:
            st.error("❌ Erreur d'authentification : Token invalide ou manquant.")
        elif response.status_code == 400:
            st.warning("⚠️ Données invalides : Veuillez dessiner un chiffre correct.")
        else:
            st.error(f"🚨 Erreur inattendue : {response.json()}")

    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Erreur de connexion à l'API : {e}")
        logger.error(f"🌐 Erreur de connexion à l'API : {e}")

    return None

def send_feedback(correct, chiffre_reel):
    """ Envoie le feedback à l’API FastAPI """
    headers = {"x-token": API_KEY}
    data = {"prediction": temp_state.prediction, "correct": correct, "chiffre_reel": chiffre_reel}

    logger.info(f"🟢 Envoi du feedback : {data}")

    try:
        response = requests.post(f"{API_URL}/feedback", json=data, headers=headers)
        logger.info(f"📡 Statut HTTP : {response.status_code}, {response.text}")

        if response.status_code == 200:
            st.success("✅ Feedback enregistré avec succès !")
        else:
            st.error(f"❌ Erreur API : {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Erreur de connexion à l’API : {e}")
        logger.error(f"🌐 Erreur de connexion à l’API : {e}")



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
            temp_state.image_data = np.array(img).tolist()
            temp_state.prediction = predict_image(img)

    if temp_state.prediction is not None:
        st.write(f"📊 **Prédiction : {temp_state.prediction}**")

        chiffre_reel = st.number_input("🔢 Indiquez le vrai chiffre", min_value=0, max_value=9, step=1, value=temp_state.prediction)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct"):
                send_feedback(1, chiffre_reel)

        with col2:
            if st.button("❌ Incorrect"):
                send_feedback(0, chiffre_reel)


elif menu == "Image aléatoire":
    st.header("🖼️ Prédiction sur une image aléatoire")

    if st.button("🎲 Charger une image aléatoire"):
        image = np.random.rand(28, 28) * 255
        temp_state.image_data = image.tolist()
        st.image(image, width=150, caption="🖼️ Image générée")
        temp_state.prediction = predict_image(image)

    if temp_state.prediction is not None:
        st.write(f"📊 **Prédiction : {temp_state.prediction}**")

        chiffre_reel = st.number_input("🔢 Indiquez le vrai chiffre", min_value=0, max_value=9, step=1, value=temp_state.prediction)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct"):
                send_feedback(1, chiffre_reel)

        with col2:
            if st.button("❌ Incorrect"):
                send_feedback(0, chiffre_reel)


elif menu == "📊 Statistiques":
    st.header("📊 Suivi des performances du modèle")

    response = requests.get(f"{API_URL}/feedback_stats", headers={"x-token": API_KEY})
    if response.status_code == 200:
        stats = response.json()
        st.write(stats)
    else:
        st.error("Impossible de récupérer les statistiques.")
