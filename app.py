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
API_KEY = os.getenv("API_KEY", "default_token")  # Récupération du token sécurisé

# Configurer les logs
logger.add("logs/streamlit_app.log", rotation="1 day", level="INFO")

# Interface Streamlit
st.title("📝 Reconnaissance de chiffres manuscrits")

menu = st.sidebar.selectbox("📌 Menu", ["Dessin", "Image aléatoire", "📊 Statistiques"])

def predict_image(image):
    """ Envoie l'image à l'API FastAPI et retourne la prédiction """
    image = np.array(image).astype("float32").flatten().tolist()  # Conversion en liste

    headers = {"x-token": API_KEY}  # Ajout du token d'authentification

    try:
        response = requests.post(f"{API_URL}/predict", json={"data": image}, headers=headers)
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

def send_feedback(image_data, prediction, correct):
    """ Envoie le feedback à l’API FastAPI """
    headers = {"x-token": API_KEY}
    data = {"image_data": str(image_data), "prediction": prediction, "correct": correct}

    try:
        response = requests.post(f"{API_URL}/feedback", json=data, headers=headers)
        if response.status_code == 200:
            st.success("✅ Feedback enregistré avec succès !")
        else:
            st.error("❌ Erreur lors de l’enregistrement du feedback.")
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Erreur de connexion à l’API : {e}")

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

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Correct"):
                        send_feedback(img.tolist(), prediction, 1)

                with col2:
                    if st.button("❌ Incorrect"):
                        send_feedback(img.tolist(), prediction, 0)

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

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Correct", key="correct_random"):
                    send_feedback(image.tolist(), prediction, 1)

            with col2:
                if st.button("❌ Incorrect", key="incorrect_random"):
                    send_feedback(image.tolist(), prediction, 0)

# Mode "Statistiques" - Voir les performances du modèle
elif menu == "📊 Statistiques":
    st.header("📊 Suivi des performances du modèle")

    response = requests.get(f"{API_URL}/feedback_stats", headers={"x-token": API_KEY})
    if response.status_code == 200:
        stats = response.json()

        st.write("### ✅ Prédictions Correctes")
        for row in stats["correct_counts"]:
            st.write(f"Chiffre {row['prediction']} : {row['count']} validations correctes")

        st.write("### ❌ Prédictions Incorrectes")
        for row in stats["incorrect_counts"]:
            st.write(f"Chiffre {row['prediction']} : {row['count']} erreurs signalées")
    else:
        st.error("Impossible de récupérer les statistiques.")
