from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import dill
from enum import Enum
from typing import List
from Src.utils import convert_explanation

class ItemType(str, Enum):
    M = "M"
    L = "L"
    H = "H"

class ItemInput(BaseModel):
    Type: ItemType = Field(..., title="The type of the item", description="The type of the item (M, L, H)", max_length=1)
    Air_temperature: float = Field(..., title="The air temperature in Kelvin", ge=295.3, le=304.5)
    Process_temperature: float = Field(..., title="The process temperature in Kelvin", ge=305.7, le=313.8)
    Rotational_speed: float = Field(..., title="The rotational speed in RPM", ge=1168, le=2886)
    Torque: float = Field(..., title="The torque in N.m", ge=3.8, le=76.6)
    Tool_wear: float = Field(..., title="The tool wear in minutes", ge=0, le=253)

class ItemOutputPredict(BaseModel):
    name: str
    proba: float

class ItemOutputExplain(BaseModel):
    item: str
    weight: float

app = FastAPI()
conv_expl = convert_explanation.explanation_convertion()

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
    """
    Returns a list of column names in the 'features' dataframe.
    """
    return features.columns.tolist()

@app.get("/features/{variable}")
def get_features(variable: str):
    """
    Retrieve the values of a specific variable from the features dataframe.

    Parameters:
    variable (str): The name of the variable/column to retrieve.

    Returns:
    dict: A dictionary containing the values of the specified variable.
          If the variable does not exist in the dataframe, an error message is returned.
    """
    # Vérifier si la colonne existe
    if variable not in features.columns:
        return {"error": f"La colonne {variable} n'existe pas"}

    # Convertir le dataframe en dictionnaire et le retourner
    return features[variable].to_dict()

@app.post('/predict', response_model=List[ItemOutputPredict])
async def predict(item: ItemInput):
    """
    Endpoint for making predictions.

    Args:
        item (Item): The input item for prediction.
        example : {
                    "Type": "M",
                    "Air_temperature": 295.3,
                    "Process_temperature": 305.7,
                    "Rotational_speed": 1168,
                    "Torque": 3.8,
                    "Tool_wear": 253
                    }

    Returns:
        dict: A dictionary containing the prediction resultwith the probabilite for each classes.
    """
    # Préparer les données pour la prédiction
    data_transformed = TransformData(item)
    
    # Faire une prédiction
    prediction = model.predict(data_transformed)

    # Retourner la prédiction
    dict_predict = [{"name":"Heat dissipation failure", "proba":prediction.tolist()[0][0]}, 
            {"name":"No failure", "proba":prediction.tolist()[0][1]},
            {"name":"Overtrain failure", "proba":prediction.tolist()[0][2]},
            {"name":"Power failure", "proba":prediction.tolist()[0][3]},
            {"name":"Tool wear failure", "proba":prediction.tolist()[0][4]}]
    return dict_predict
    #return {'prediction': prediction.tolist()}

@app.post('/explain', response_model=List[ItemOutputExplain])
async def explain(item: ItemInput):
    """
    Endpoint for explaining an item.
    
    Args:
        item (Item): The item to be explained.
        example : {
                "Type": "M",
                "Air_temperature": 295.3,
                "Process_temperature": 305.7,
                "Rotational_speed": 1168,
                "Torque": 3.8,
                "Tool_wear": 253
                }
    
    Returns:
        dict: A dictionary containing the explanation.
    """
    # Préparer les données pour la prédiction
    data_transformed = TransformData(item)
    
    # Faire une prédiction
    prediction = model.predict(data_transformed)

    # Expliquer la prédiction
    exp = loaded_explainer.explain_instance(data_transformed[0], model.predict, num_features=prediction.shape[1])
    
    # Retourner la prédiction
    #print(conv_expl.convert_data_explanation(exp.as_list(), ct_X_))
    dict_explain = [{"item":x[0], "weight":x[1]} for x in conv_expl.convert_data_explanation(exp.as_list(), ct_X_)]
    return dict_explain
    #return {'prediction': prediction.tolist(), 'explanation': conv_expl.convert_data_explanation(exp.as_list(), ct_X_)}



def TransformData(item: ItemInput):
    """
    Transforms the given item data into a format suitable for prediction.

    Args:
        item (Item): The item containing the data to be transformed.

    Returns:
        tab1 (Numpy): The transformed data in a Numpy format.
    """

    if item.Type == 'M':
        tmp = 'M'
    elif item.Type == 'L':
        tmp = 'L'
    elif item.Type == 'H':
        tmp = 'H'
    else:
        tmp = 'M'
    data = np.array([tmp,
                    item.Air_temperature,
                    item.Process_temperature, 
                    item.Rotational_speed, 
                    item.Torque, 
                    item.Tool_wear, 
                    item.Process_temperature-item.Air_temperature,
                    item.Torque*item.Rotational_speed])
    data = pd.DataFrame(data.reshape(1, -1), columns=ct_X_.feature_names_in_)
    tab1 = ct_X_.transform(data)
    return tab1