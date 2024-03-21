# Test the main.py file with the following test_main.py file:
import os
import sys
from typing import List
sys.path.insert(0, os.getcwd())
import pytest
from fastapi.testclient import TestClient
from main import app, ItemInput


client = TestClient(app)

def test_get_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "hello world !"

def test_post_predict():
    item = ItemInput(Type="M", 
                     Air_temperature=295.3, 
                     Process_temperature=305.7, 
                     Rotational_speed=1168, 
                     Torque=3.8, 
                     Tool_wear=253
                     #Difference_temperature=1, Power=1
                     )
    response = client.post("/predict", json=item.dict())
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 5
    assert isinstance(response.json()[0]['proba'], float)
    assert isinstance(response.json()[0]['name'], str)
    assert all(isinstance(item, dict) for item in response.json())

def test_post_explain():
    item = ItemInput(Type="M",
                     Air_temperature=295.3,
                     Process_temperature=305.7,
                     Rotational_speed=1168,
                     Torque=3.8,
                     Tool_wear=253
                     #Difference_temperature=1, Power=1
                     )    
    response = client.post("/explain", json=item.dict())
    assert response.status_code == 200
    print(10*"!")
    print(response.json())
    assert isinstance(response.json(), list)
    assert len(response.json()) == 5
    assert isinstance(response.json()[0]['weight'], float)
    assert isinstance(response.json()[0]['item'], str)
    assert all(isinstance(item, dict) for item in response.json())
    
    
