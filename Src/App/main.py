from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import dill

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

import os
import sys


def move_to_project_dir():
    _st = os.getcwd()

    _k = 5
    if os.getenv('NAME_PROJECT') is None:
        os.environ['NAME_PROJECT'] = "Pred_maint_projects"

    _project_name = os.getenv('NAME_PROJECT')
    if (_project_name in _st):
        while (_st.split(os.sep)[-1]!=_project_name):
            os.chdir("..")
            _st = os.getcwd()
            _k -= 1
            if _k < 1:
                break
    else:
        print(f'current dir: {os.getcwd()}')
        print("not found")

move_to_project_dir()

sys.path.insert(0, os.getcwd())

from Src.config import configuration
cfg = configuration.config_manager().get_path()

# Chargement du modèle
model = load_model('.'+os.sep+cfg.config_path.model_dir+os.sep+'best_model.h5')

# Chargement du transformer
file_transformer = '.'+os.sep+cfg.config_path.work_dir+os.sep+'transformer.pkl'
ct_X_ = joblib.load(file_transformer)

# Chargement de l'explainer
file_explainer = '.'+os.sep+cfg.config_path.work_dir+os.sep+'explainer.pkl'
with open(file_explainer, 'rb') as f:
    loaded_explainer = dill.load(f)

# Chargement du fichier csv contenant les caracteristiques des données
file_features = '.'+os.sep+cfg.config_path.work_dir+os.sep+'dataset_caracteristics.csv'
features = pd.read_csv(file_features, index_col=0).fillna(value=0)

@app.get('/')
def get_index():
    return 'hello world !'

@app.get("/list_features")
def list_features():
    return features.columns.tolist()

@app.get("/features/{variable}")
def get_features(variable: str):
    # Vérifier si la colonne existe
    if variable not in features.columns:
        return {"error": f"La colonne {variable} n'existe pas"}
    print(10*'_')
    print(features[variable].to_dict())
    # Convertir le dataframe en dictionnaire et le renvoyer
    return features[variable].to_dict()

@app.post('/predict')
async def predict(item: Item):
    # Préparer les données pour la prédiction
    data_transformed = TransformData(item)
    
    # Faire une prédiction
    prediction = model.predict(data_transformed)
    
    # Retourner la prédiction
    return {'prediction': prediction.tolist()}

@app.post('/predict_explain')
async def explain(item: Item):
    # Préparer les données pour la prédiction
    data_transformed = TransformData(item)
    
    # Faire une prédiction
    prediction = model.predict(data_transformed)

    # Expliquer la prédiction
    exp = loaded_explainer.explain_instance(data_transformed[0], model.predict, num_features=prediction.shape[1])
    
    # Retourner la prédiction
    return {'prediction': prediction.tolist(), 'explanation': exp.as_list()}

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