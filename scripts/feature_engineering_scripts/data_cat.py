import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('data/baselines/stage3/train.csv')
df_test = pd.read_csv('data/baselines/stage3/test.csv')

df_train = df_train.drop(df_train[(df_train.sub_area == 'Poselenie Klenovskoe') | (df_train.sub_area == 'Poselenie Mihajlovo-Jarcevskoe') | (df_train.sub_area == 'Poselenie Shhapovskoe')].index)

df_train = df_train.drop(('timestamp'), axis=1)

df_test = df_test.drop(('timestamp'), axis=1)

os.mkdir("data/baselines/stage4")

df_train.to_csv('data/baselines/stage4/train.csv', index=False)
df_test.to_csv('data/baselines/stage4/test.csv', index=False)