import os
import sys
import yaml

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path: str, obj: object):
    """Saves an object to a specified file path.

    This function serializes the provided object using pickle and 
    saves it to the specified file. If the directory does not exist, 
    it will be created.

    Args:
        file_path (str): The path where the object should be saved, including the filename.
        obj (object): The object to be saved.

    Raises:
        CustomException: If an error occurs during saving, 
                         an exception with the error message and traceback will be raised.

    Example:
        save_object('artifacts/model.pkl', trained_model)
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """Evaluates multiple models using the provided training and test data.

    This function trains each model on the training data, performs hyperparameter 
    tuning using GridSearchCV, and calculates the R² score for both training 
    and test sets.

    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target data.
        models (dict): A dictionary of models to be evaluated.
        param (dict): A dictionary of hyperparameters for each model.

    Returns:
        dict: A dictionary containing the test R² scores for each model.

    Raises:
        CustomException: If an error occurs during evaluation, 
                         an exception with the error message and traceback will be raised.

    Example:
        report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a serialized object from a specified file path.

    This function deserializes an object from the specified file using pickle.

    Args:
        file_path (str): The path from where the object should be loaded, including the filename.

    Returns:
        object: The deserialized object loaded from the file.

    Raises:
        CustomException: If an error occurs during loading, 
                         an exception with the error message and traceback will be raised.

    Example:
        model = load_object('artifacts/model.pkl')
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_config(file_path):
    """
    Load model configurations from a YAML file.

    This function reads a YAML file containing model definitions and their
    hyperparameters. It dynamically creates instances of the specified models
    using the provided parameters.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        tuple: A tuple containing:
            - models (dict): A dictionary of model instances, where keys are 
              model names and values are instantiated model objects.
            - params (dict): A dictionary of hyperparameters for each model, 
              where keys are model names and values are dictionaries of parameters.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    models = {}
    params = {}
    for model_name, model_info in config['models'].items():
        model_class = globals().get(model_name)  # globals to get model class
        if model_class:
            models[model_name] = model_class(**{k: v for k, v in model_info.items() if k != 'params'})
            params[model_name] = model_info['params']
    return models, params