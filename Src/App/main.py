from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import numpy as np

class Item(BaseModel):
    Air_temperature: int
    Process_temperature: int
    Rotational_speed: int
    Torque: int
    Tool_wear: int
    Difference_temperature: int
    Power: int
    Type_H: bool
    Type_L: bool
    Type_M: bool


api = FastAPI()

@api.get('/')
def get_index():
    #return {'data': 'hello world'}
    return 'hello world'


@api.post('/predict')
async def predict(item: Item):
    # Préparer les données pour la prédiction
    data = np.array(item.data)
    
    # Faire une prédiction
    prediction = model.predict(data)
    
    # Retourner la prédiction
    return {'prediction': prediction.tolist()}