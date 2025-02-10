import streamlit as st
import requests
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_URL = "https://webappfastapi-faazh4eya0gnamew.westeurope-01.azurewebsites.net/predict"

st.title("Reconnaissance de chiffres manuscrits")

menu = st.sidebar.selectbox("Menu", ["Dessin", "Image aléatoire"])

def predict_image(image):
    image = np.array(image).astype("float32").flatten().tolist()  # Conversion en liste
    response = requests.post(API_URL, json={"data": image})
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return "Erreur API"

if menu == "Dessin":
    st.header("Dessinez un chiffre")
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

    if st.button("Prédire"):
        if canvas.image_data is not None:
            img = Image.fromarray((canvas.image_data[:, :, 0]).astype("uint8"))
            img = img.resize((28, 28))
            prediction = predict_image(img)
            st.write(f"Prédiction : {prediction}")

elif menu == "Image aléatoire":
    st.header("Prédiction sur une image aléatoire")
    if st.button("Charger une image aléatoire"):
        index = np.random.randint(0, 1000)
        image = np.random.rand(28, 28) * 255  # Remplace ceci par tes vraies images
        st.image(image, width=150, caption="Image générée")
        prediction = predict_image(image)
        st.write(f"Prédiction : {prediction}")
