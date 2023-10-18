import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('data/baselines/stage2/train.csv')
df_test = pd.read_csv('data/baselines/stage2/test.csv')

df_copy_train = df_train.copy()
df_copy_test = df_test.copy()

df_copy_train.build_year = df_copy_train['build_year'].fillna(
    round(df_copy_train.groupby(['sub_area', 'max_floor'])['build_year'].transform('mean'), 0))
df_copy_train.build_year = df_copy_train['build_year'].fillna(
    round(df_copy_train.groupby(['sub_area'])['build_year'].transform('mean'), 0))
df_copy_test.build_year = df_copy_test['build_year'].fillna(
    round(df_copy_train.groupby(['sub_area', 'max_floor'])['build_year'].transform('mean'), 0))

df_copy_test.loc[(df_copy_test.id==2012),'full_sq'] = 40
df_copy_test.loc[(df_copy_test.id==2012),'kitch_sq'] = 0
df_copy_test.loc[(df_copy_test.id==1416),'full_sq'] = 53
df_copy_test.loc[(df_copy_test.id==1416),'kitch_sq'] = df_copy_test.full_sq - df_copy_test.life_sq
df_copy_test.loc[(df_copy_test.id==5752),'full_sq'] = 45
df_copy_test.loc[(df_copy_test.id==5752),'kitch_sq'] = df_copy_test.full_sq - df_copy_test.life_sq
df_copy_test.loc[(df_copy_test.id==4441),'full_sq'] = 40
df_copy_test.loc[(df_copy_test.id==4441),'kitch_sq'] = df_copy_test.full_sq - df_copy_test.life_sq
df_copy_test.loc[(df_copy_test.id==3061),'full_sq'] = 54
df_copy_test.loc[(df_copy_test.id==3061),'kitch_sq'] = df_copy_test.full_sq - df_copy_test.life_sq
df_copy_test.loc[(df_copy_test.id==22798),'full_sq'] = 50
df_copy_test.loc[(df_copy_test.id==22798),'life_sq'] = df_copy_test.full_sq - df_copy_test.kitch_sq
df_copy_test.loc[(df_copy_test.id==25890),'full_sq'] = 67
df_copy_test.loc[(df_copy_test.id==25890),'life_sq'] = df_copy_test.full_sq - df_copy_test.kitch_sq

df_copy_test['kitch_sq'] = df_copy_test['kitch_sq'].fillna(round(
    df_copy_train.groupby(['sub_area','build_year'])['kitch_sq'].transform('mean'),0))

df_copy_test.loc[(df_copy_test.life_sq.isna()),'life_sq'] = df_copy_test.full_sq - df_copy_test.kitch_sq

df_copy_train.state = df_copy_train['state'].fillna(round(df_copy_train.groupby(['sub_area','build_year'])['state'].transform('mean'), 0))

df_copy_train.state = df_copy_train['state'].fillna(round(df_copy_train.groupby(['sub_area','max_floor'])['state'].transform('mean'), 0))
df_copy_train.state = df_copy_train['state'].fillna(round(df_copy_train.groupby(['build_year'])['state'].transform('mean'), 0))

df_copy_test.state = df_copy_test['state'].fillna(
    round(df_copy_train.groupby(['sub_area','build_year'])['state'].transform('mean'), 0))

df_copy_test.material = df_copy_test['material'].fillna(
    round(df_copy_train.groupby(['sub_area','build_year'])['state'].transform('mean'), 0))

df_copy_test.loc[(df_copy_test.num_room.isna()) & (df_copy_test.life_sq < 30),'num_room'] = 1
df_copy_test.loc[(df_copy_test.num_room.isna()) & (df_copy_test.life_sq < 45),'num_room'] = 2
df_copy_test.loc[(df_copy_test.num_room.isna()) & (df_copy_test.life_sq < 75),'num_room'] = 3
df_copy_test.loc[(df_copy_test.num_room.isna()) & (df_copy_test.life_sq < 100),'num_room'] = 4
df_copy_test.loc[(df_copy_test.num_room.isna()) & (df_copy_test.life_sq < 150),'num_room'] = 5
df_copy_test.loc[(df_copy_test.num_room.isna()),'num_room'] = 6

df_train = df_copy_train.copy()
df_test = df_copy_test.copy()

os.mkdir("data/baselines/stage3")

df_train.to_csv('data/baselines/stage3/train.csv', index=False)
df_test.to_csv('data/baselines/stage3/test.csv', index=False)

