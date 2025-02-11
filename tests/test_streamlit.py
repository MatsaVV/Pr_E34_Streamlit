import pytest
import requests
import numpy as np
from PIL import Image

# URL de l'API déployée sur Azure
API_URL = "https://webappfastapi-faazh4eya0gnamew.westeurope-01.azurewebsites.net/predict"

@pytest.fixture
def sample_image():
    """Crée une image de test de 28x28 pixels"""
    img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    return img

def test_api_response(sample_image):
    """Vérifie que l'API répond correctement"""
    data = sample_image.flatten().tolist()
    response = requests.post(API_URL, json={"data": data})
    assert response.status_code == 200, f"Erreur API : {response.text}"

def test_api_prediction_format(sample_image):
    """Vérifie que la réponse contient une clé 'prediction' avec un entier"""
    data = sample_image.flatten().tolist()
    response = requests.post(API_URL, json={"data": data})
    assert "prediction" in response.json(), "La réponse ne contient pas 'prediction'"
    assert isinstance(response.json()["prediction"], int), "La prédiction n'est pas un entier"

def test_api_with_black_image():
    """Teste la prédiction sur une image noire"""
    black_img = np.zeros((28, 28), dtype=np.uint8)
    data = black_img.flatten().tolist()
    response = requests.post(API_URL, json={"data": data})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

def test_api_with_white_image():
    """Teste la prédiction sur une image blanche"""
    white_img = np.ones((28, 28), dtype=np.uint8) * 255
    data = white_img.flatten().tolist()
    response = requests.post(API_URL, json={"data": data})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
