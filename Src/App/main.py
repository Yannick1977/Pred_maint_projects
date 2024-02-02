from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

class Item(BaseModel):
    Type: str
    Air_temperature: int
    Process_temperature: int
    Rotational_speed: int
    Torque: int
    Tool_wear: int
    Difference_temperature: int
    Power: int


app = FastAPI()

# Charger le modèle Keras
import os
import sys
os.chdir('../..')
sys.path.insert(0, os.getcwd())

from Src.config import configuration
cfg = configuration.config_manager().get_path()

# Chargement du modèle
model = load_model('./'+cfg.config_path.model_dir+'/best_model.keras')

# Chargement du transformer
file_transformer = './'+cfg.config_path.work_dir+'/transformer.pkl'
ct_X_ = joblib.load(file_transformer)

@app.get('/')
def get_index():
    return 'hello world !'

@app.post('/predict')
async def predict(item: Item):
    # Préparer les données pour la prédiction
    data_transformed = TransformData(item)
    
    # Faire une prédiction
    prediction = model.predict(data_transformed)
    
    # Retourner la prédiction
    return {'prediction': prediction.tolist()}

@app.post('/test')
async def test(item: Item):
    # Préparer les données pour la prédiction
    data_transformed = TransformData(item)
    return {'test': data_transformed.tolist()}


def TransformData(item: Item):
    """
    Transforms the given item data into a format suitable for prediction.

    Args:
        item (Item): The item containing the data to be transformed.

    Returns:
        tab1 (Numpy): The transformed data in a Numpy format.
    """
    data = np.array([item.Type,
                    item.Air_temperature,
                    item.Process_temperature, 
                    item.Rotational_speed, 
                    item.Torque, 
                    item.Tool_wear, 
                    item.Difference_temperature,
                    item.Power])
    data = pd.DataFrame(data.reshape(1, -1), columns=ct_X_.feature_names_in_)
    tab1 = ct_X_.transform(data)
    return tab1