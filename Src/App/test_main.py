import pytest
from fastapi.testclient import TestClient
from main import app, Item  # Assurez-vous que votre fichier s'appelle main.py

client = TestClient(app)

def test_get_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "hello world !"

def test_post_predict():
    item = Item(Type="M", Air_temperature=1, Process_temperature=1, Rotational_speed=1, Torque=1, Tool_wear=1, Difference_temperature=1, Power=1)
    response = client.post("/predict", json=item.dict())
    assert response.status_code == 200
    assert "prediction" in response.json()