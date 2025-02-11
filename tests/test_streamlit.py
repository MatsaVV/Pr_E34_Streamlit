import pytest
import requests
import numpy as np
import os
from dotenv import load_dotenv

# Charger la variable d'environnement pour le token API
load_dotenv()
API_URL = "https://webappfastapi-faazh4eya0gnamew.westeurope-01.azurewebsites.net/predict"
API_KEY = os.getenv("API_KEY", "default_token")

@pytest.fixture
def sample_image():
    """Crée une image de test de 28x28 pixels"""
    img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    return img

def send_request(data):
    """Envoie une requête avec gestion des erreurs"""
    headers = {"x-token": API_KEY}
    response = requests.post(API_URL, json={"data": data}, headers=headers)
    return response

def test_api_prediction_format(sample_image):
    """Vérifie que la réponse contient une clé 'prediction' avec un entier"""
    response = send_request(sample_image.flatten().tolist())
    assert "prediction" in response.json(), "La réponse ne contient pas 'prediction'"
    assert isinstance(response.json()["prediction"], int), "La prédiction n'est pas un entier"

def test_api_with_black_image():
    """Teste la prédiction sur une image noire"""
    black_img = np.zeros((28, 28), dtype=np.uint8)
    response = send_request(black_img.flatten().tolist())
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

def test_api_with_white_image():
    """Teste la prédiction sur une image blanche"""
    white_img = np.ones((28, 28), dtype=np.uint8) * 255
    response = send_request(white_img.flatten().tolist())
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

def test_api_with_invalid_input():
    """Teste l'API avec une entrée invalide (mauvaise taille)"""
    invalid_data = [0] * 100  # Seulement 100 pixels au lieu de 784
    headers = {"x-token": API_KEY}

    response = requests.post(API_URL, json={"data": invalid_data}, headers=headers)

    assert response.status_code == 400, "L'API aurait dû renvoyer 400 pour une donnée invalide"

def test_api_with_no_token():
    """Teste une requête sans token (devrait renvoyer 401 ou 403)"""
    data = np.random.randint(0, 256, (28, 28)).flatten().tolist()
    response = requests.post(API_URL, json={"data": data})  # Pas de headers !

    assert response.status_code in [401, 403], f"L'API aurait dû renvoyer 401 ou 403, mais a renvoyé {response.status_code}"
