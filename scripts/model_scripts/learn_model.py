import os
import pandas as pd
import numpy as np
import pickle
import yaml
import joblib
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('data/baselines/stage4/train.csv')
df_test = pd.read_csv('data/baselines/stage4/test.csv')

X = df_train.drop(('price_doc'), axis=1)
X = X.drop(('id'), axis=1)
y = df_train['price_doc']

params_tts = yaml.safe_load(open("params.yaml"))["split"]
test_size = params_tts["test_size"]
random_state = params_tts["random_state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

params = yaml.safe_load(open("params.yaml"))["train"]
cat_features = params["cat_features"]
learning_rate = params["learning_rate"]
iterations = params["iterations"]
loss_function = params["loss_function"]
random_seed = params["seed"]

from catboost import CatBoostRegressor

model = CatBoostRegressor(cat_features=cat_features, learning_rate=learning_rate,iterations=iterations,loss_function=loss_function,random_seed=random_seed)
print("start fit")
model.fit(X,y,verbose=0)
print("end fit")
os.mkdir("data/final")

with open ('data/final/model.pkl', 'wb') as file:
    joblib.dump(model, file)