import os
import pandas as pd
import numpy as np
import pickle
import yaml
import joblib
import json


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df_train = pd.read_csv('data/baselines/stage4/train.csv')
df_test = pd.read_csv('data/baselines/stage4/test.csv')

X = df_train.drop(('price_doc'), axis=1)
X = X.drop(('id'), axis=1)
y = df_train['price_doc']

params_tts = yaml.safe_load(open("params.yaml"))["split"]
test_size = params_tts["test_size"]
random_state = params_tts["random_state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

model = joblib.load('data/final/model.pkl')

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(mae)
output = {"score": mae} 

with open(f"score/evaluate.json", "w") as write_file:
            json.dump(output, write_file)