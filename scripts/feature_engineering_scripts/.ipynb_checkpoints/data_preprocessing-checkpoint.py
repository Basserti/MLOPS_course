import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('data/raw/train.csv')
df_test = pd.read_csv('data/raw/test.csv')
df_macro = pd.read_csv('data/raw/macro.csv')
df_all = data = pd.concat([df_train, df_test]).reset_index(drop=True)

df_macro = df_macro[df_macro.timestamp.isin(df_all.timestamp.unique())]

df_train.timestamp = pd.to_datetime(df_train.timestamp)
df_test.timestamp = pd.to_datetime(df_test.timestamp)
df_macro.timestamp = pd.to_datetime(df_macro.timestamp)

nums = '0|1|2|3|4|5|6|7|8|9'
df_train[df_train.build_year.str.contains(nums) == False]

df_train.loc[df_train.build_year.str.contains(nums) == False,'build_year'] = 0

df_train.build_year = pd.to_numeric(df_train.build_year)
df_test.build_year = pd.to_numeric(df_test.build_year)


df_train.loc[df_train['build_year'] == 71, 'build_year'] = 1971
df_train.loc[df_train['build_year'] == 215, 'build_year'] = 2015
df_train.loc[df_train['build_year'] == 20, 'build_year'] = 2000
df_train.loc[df_train['build_year'] == 20052009, 'build_year'] = 2009
df_test.loc[df_test['build_year'] >= 2023, 'build_year'] = 1965
df_train.loc[df_train['build_year'] == 3, 'build_year'] = 2003
df_test.loc[df_test['build_year'] == 3, 'build_year'] = 2003

df_train.drop(df_train[(df_train.index == 339)].index, inplace=True)
df_train.loc[(df_train.sub_area == 'Ochakovo-Matveevskoe') &
             (df_train.build_year == 0),'build_year'] = 2008 
df_train.loc[(df_train.sub_area == 'Ochakovo-Matveevskoe') &
             (df_train.build_year == 1),'build_year'] = round(df_train[(df_train.sub_area == 'Ochakovo-Matveevskoe') &
             (df_train.material == 1)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Novofedorovskoe') &
             ((df_train.build_year == 1) | (df_train.build_year == 0)) 
             & ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0)), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Novofedorovskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Novofedorovskoe') &
             (df_train.build_year == 0), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Novofedorovskoe') &
             (df_train.material == 6)&
             (df_train.build_year != 0)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Novofedorovskoe') &
             ((df_test.build_year < 10)) 
             & ((df_test.kitch_sq == 10)|(df_test.kitch_sq == 1)),'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Novofedorovskoe') &
             ((df_train.kitch_sq == 10)|(df_train.kitch_sq == 1))&
             (df_train.build_year > 0)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Tverskoe') &
             ((df_train.build_year < 10)) 
             & ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0)  | 
              (df_train.kitch_sq == 10)), 'build_year'] = round(df_train[(df_train.sub_area == 'Tverskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Tverskoe') &
             ((df_test.build_year < 10)) 
             & ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0)
               ), 'build_year'] = round(df_train[(df_train.sub_area == 'Tverskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0) 

df_train.loc[(df_train.sub_area == 'Veshnjaki') &
             ((df_train.kitch_sq == 6) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Veshnjaki') &
             ((df_train.kitch_sq == 6) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Veshnjaki') &
             ((df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Veshnjaki') &
             ((df_train.kitch_sq == 0))].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Filimonkovskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Filimonkovskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Filimonkovskoe') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Filimonkovskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Shherbinka') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Shherbinka') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Shherbinka') &
             ((df_train.kitch_sq == 10) )&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Shherbinka') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Shherbinka') &
             ((df_test.kitch_sq == 1)| (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Shherbinka') &
             ((df_train.kitch_sq == 1)| (df_train.kitch_sq == 0))].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Shherbinka') &
             ((df_test.kitch_sq == 9))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Shherbinka') &
             ((df_train.kitch_sq == 9))].build_year.mean(),0)


df_train.loc[(df_train.sub_area == 'Poselenie Pervomajskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Pervomajskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Pervomajskoe') &
             ((df_train.kitch_sq == 11) | (df_train.kitch_sq == 12))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Pervomajskoe') &
             ((df_train.kitch_sq == 12) | (df_train.kitch_sq == 11))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Desjonovskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Desjonovskoe') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Desjonovskoe') &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Desjonovskoe') &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Desjonovskoe') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Desjonovskoe') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Moskovskij') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year']  = round(df_train[(df_train.sub_area == 'Poselenie Moskovskij') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Moskovskij') &
             ((df_train.kitch_sq == 10) | (df_train.kitch_sq == 17)
             | (df_train.kitch_sq == 4))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Moskovskij') &
             ((df_train.kitch_sq == 10) | (df_train.kitch_sq == 17)
             | (df_train.kitch_sq == 4))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Moskovskij') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Moskovskij') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Moskovskij') &
             ((df_test.kitch_sq == 12) | (df_test.kitch_sq == 15))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Moskovskij') &
             ((df_train.kitch_sq == 12) | (df_train.kitch_sq == 15))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Zapadnoe Degunino') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Zapadnoe Degunino') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Zapadnoe Degunino') &
             ((df_train.kitch_sq == 10) | (df_train.kitch_sq == 43))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Zapadnoe Degunino') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Zapadnoe Degunino') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Zapadnoe Degunino') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Solncevo') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Solncevo') &
             ((df_train.kitch_sq == 1)| (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Solncevo') &
             ((df_train.kitch_sq == 10) | (df_train.kitch_sq == 14))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Solncevo') &
             ((df_train.kitch_sq == 10)| (df_train.kitch_sq == 14))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Solncevo') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year']= round(df_train[(df_train.sub_area == 'Solncevo') &
             ((df_train.kitch_sq == 1)| (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Solncevo') &
             ((df_test.kitch_sq == 12))&
             (df_test.build_year < 10), 'build_year']= round(df_train[(df_train.sub_area == 'Solncevo') &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Vnukovskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Vnukovskoe') &
             ((df_train.kitch_sq == 1)| (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Vnukovskoe') &
             ((df_train.kitch_sq == 10)| (df_train.kitch_sq == 12)
             | (df_train.kitch_sq == 20))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Vnukovskoe') &
             ((df_train.kitch_sq == 10)| (df_train.kitch_sq == 12)
             | (df_train.kitch_sq == 20))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Vnukovskoe') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10),'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Vnukovskoe') &
             ((df_train.kitch_sq == 1)| (df_train.kitch_sq == 0))].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Vnukovskoe') &
             ((df_test.kitch_sq == 10))&
             (df_test.build_year < 10),'build_year'] =round(df_train[(df_train.sub_area == 'Poselenie Vnukovskoe') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 0)| (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 9))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 9)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 10)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 8))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 8)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 11))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 11)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 12)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Sosenskoe') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 0)| (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Sosenskoe') &
             ((df_test.kitch_sq == 9))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 9)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Sosenskoe') &
             ((df_test.kitch_sq == 11))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 11)) &
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Sosenskoe') &
             ((df_test.kitch_sq == 12))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Sosenskoe') &
             ((df_train.kitch_sq == 12)) &
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Krasnopahorskoe') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Krasnopahorskoe') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Rogovskoe') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1)
             | (df_train.kitch_sq == 8))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Rogovskoe') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1)
              | (df_train.kitch_sq == 8))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Poselenie Rogovskoe') &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year']= round(df_train[(df_train.sub_area == 'Poselenie Rogovskoe') &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Nekrasovka') &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Nekrasovka') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Nekrasovka') &
             ((df_train.kitch_sq == 12) | (df_train.kitch_sq == 13))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Nekrasovka') &
             ((df_train.kitch_sq == 13))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Nekrasovka') &
             ((df_test.kitch_sq == 1) | (df_test.kitch_sq == 0)| (df_test.kitch_sq == 10)| (df_test.kitch_sq == 12))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Nekrasovka') &
             ((df_train.kitch_sq == 0) | (df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Donskoe') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Donskoe') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Voskresenskoe') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1)|
             (df_train.kitch_sq == 12))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Voskresenskoe') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1)|
             (df_train.kitch_sq == 12))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Poselenie Voskresenskoe') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1)|
             (df_train.kitch_sq == 12))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Poselenie Voskresenskoe') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1)|
             (df_train.kitch_sq == 12))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Krjukovo') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Krjukovo') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Juzhnoe Butovo') &
             ((df_train.kitch_sq == 1)|(df_train.kitch_sq == 7)|
             (df_train.kitch_sq == 8)|(df_train.kitch_sq == 10))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Juzhnoe Butovo') &
             ((df_train.kitch_sq == 1)|(df_train.kitch_sq == 7)|
             (df_train.kitch_sq == 8)|(df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 19))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 19))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 15))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 15))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 18))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 18))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Troickij okrug') &
             ((df_test.kitch_sq == 12))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Troickij okrug') &
             ((df_test.kitch_sq == 13))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Troickij okrug') &
             ((df_train.kitch_sq == 13))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Mozhajskoe') &
             ((df_train.num_room == 2))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == 'Mozhajskoe') &
             ((df_train.num_room == 2))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == "Tekstil'shhiki") &
             ((df_train.kitch_sq == 5))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Tekstil'shhiki") &
             ((df_train.kitch_sq == 5))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Tekstil'shhiki") &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Tekstil'shhiki") &
             ((df_train.kitch_sq == 1) | (df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == "Teplyj Stan") &
             ((df_train.kitch_sq == 1) )&
             (df_train.build_year < 10), 'build_year'] = 2014
df_train.loc[(df_train.sub_area == "Teplyj Stan") &
             ((df_train.kitch_sq == 6) )&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Teplyj Stan") &
             ((df_train.kitch_sq == 6) )&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == "Novogireevo") &
             ((df_train.kitch_sq == 0) )&
             (df_train.build_year < 10), 'build_year'] = 2014
df_train.loc[(df_train.sub_area == "Novogireevo") &
             ((df_train.kitch_sq == 5) )&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Novogireevo") &
             ((df_train.kitch_sq == 5) )&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Mitino') &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Mitino") &
             ((df_train.kitch_sq == 0)|(df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Mitino') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Mitino") &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Danilovskoe') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = 2012
df_train.loc[(df_train.sub_area == 'Danilovskoe') &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Danilovskoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Strogino') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Strogino") &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == 'Strogino') &
             ((df_test.kitch_sq == 0)|(df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Strogino") &
             ((df_train.kitch_sq == 1)|(df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Jasenevo') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Jasenevo") &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Ramenki') &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] =round(df_train[(df_train.sub_area == "Ramenki") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Obruchevskoe') &
             ((df_train.kitch_sq == 1)|(df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Obruchevskoe") &
             ((df_train.kitch_sq == 1)|(df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Presnenskoe') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = 2001
df_test.loc[(df_test.sub_area == 'Presnenskoe') &
             ((df_test.kitch_sq == 20))&
             (df_test.build_year < 10), 'build_year'] = 2013

df_train.loc[(df_train.sub_area == 'Novo-Peredelkino') &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Novo-Peredelkino") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Chertanovo Severnoe') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] =round(df_train[(df_train.sub_area == "Chertanovo Severnoe") &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Nagatinskij Zaton') &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] =round(df_train[(df_train.sub_area == "Nagatinskij Zaton") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Nagatinskij Zaton') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] =round(df_train[(df_train.sub_area == "Nagatinskij Zaton") &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Severnoe Tushino') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] =round(df_train[(df_train.sub_area == "Severnoe Tushino") &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Prospekt Vernadskogo') &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Prospekt Vernadskogo") &
             ((df_train.kitch_sq == 0))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[(df_train.sub_area == 'Beskudnikovskoe') &
             ((df_train.kitch_sq == 9))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Beskudnikovskoe") &
             ((df_train.kitch_sq == 9))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Akademicheskoe') &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Akademicheskoe") &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Birjulevo Vostochnoe') &
             ((df_train.kitch_sq == 9)|(df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Birjulevo Vostochnoe") &
             ((df_train.kitch_sq == 9)|(df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == 'Brateevo') &
             ((df_train.kitch_sq == 8)|(df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Brateevo") &
             ((df_train.kitch_sq == 8)|(df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Chertanovo Central'noe") &
             ((df_train.kitch_sq == 5))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Chertanovo Central'noe") &
             ((df_train.kitch_sq == 5))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Chertanovo Central'noe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Chertanovo Central'noe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Hovrino") &
             ((df_train.kitch_sq == 9))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Hovrino") &
             ((df_train.kitch_sq == 9))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Kapotnja") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Kapotnja") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Krylatskoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Krylatskoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Levoberezhnoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Levoberezhnoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Mar'ino") &
             ((df_train.kitch_sq == 7))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Mar'ino") &
             ((df_train.kitch_sq == 7))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Nagornoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Nagornoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Orehovo-Borisovo Severnoe") &
             ((df_train.kitch_sq == 1)|(df_train.kitch_sq == 8)|(df_train.kitch_sq == 14))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Orehovo-Borisovo Severnoe") &
             ((df_train.kitch_sq == 8))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Otradnoe") &
             ((df_train.kitch_sq == 6))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Otradnoe") &
             ((df_train.kitch_sq == 6))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Pokrovskoe Streshnevo") &
             ((df_train.kitch_sq == 11)|(df_train.kitch_sq == 5))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Pokrovskoe Streshnevo") &
             ((df_train.kitch_sq == 11)|(df_train.kitch_sq == 5))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Pokrovskoe Streshnevo") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Pokrovskoe Streshnevo") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Severnoe Butovo") &
             ((df_train.kitch_sq == 8))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Severnoe Butovo") &
             ((df_train.kitch_sq == 8))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Severnoe Medvedkovo") &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Severnoe Medvedkovo") &
             ((df_train.kitch_sq == 12))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Timirjazevskoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Timirjazevskoe") &
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Vojkovskoe") &
             ((df_train.kitch_sq == 8))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Vojkovskoe") &
             ((df_train.kitch_sq == 8))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Vyhino-Zhulebino") &
             ((df_train.kitch_sq == 6))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Vyhino-Zhulebino") &
             ((df_train.kitch_sq == 6))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_train.loc[(df_train.sub_area == "Zjuzino") &
             ((df_train.kitch_sq == 5))&
             (df_train.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Zjuzino") &
             ((df_train.kitch_sq == 5))&
             (df_train.build_year > 10)].build_year.mean(),0)

df_test.loc[(df_test.sub_area == "Danilovskoe") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Danilovskoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Juzhnoe Medvedkovo") &
             ((df_test.kitch_sq == 10))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Juzhnoe Medvedkovo") &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Juzhnoe Tushino") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Juzhnoe Tushino") &
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Krjukovo") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Krjukovo") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Kuncevo") &
             ((df_test.kitch_sq == 9))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Kuncevo") &
             ((df_train.kitch_sq == 9))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Levoberezhnoe") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Levoberezhnoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Mar'ina Roshha") &
             ((df_test.kitch_sq == 1)|(df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Mar'ina Roshha") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Mitino") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Mitino") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Nagatinskij Zaton") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Nagatinskij Zaton") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Pokrovskoe Streshnevo") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Pokrovskoe Streshnevo") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Poselenie Pervomajskoe") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Poselenie Pervomajskoe") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Poselenie Voskresenskoe") &
             ((df_test.kitch_sq == 1)|(df_test.kitch_sq == 11))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Poselenie Voskresenskoe") &
             ((df_train.kitch_sq == 11)|(df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Severnoe Butovo") &
             ((df_test.kitch_sq == 10))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Severnoe Butovo") &
             ((df_train.kitch_sq == 10))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Severnoe Tushino") &
             ((df_test.kitch_sq == 1))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Severnoe Tushino") &
             ((df_train.kitch_sq == 1))&
             (df_train.build_year > 10)].build_year.mean(),0)
df_test.loc[(df_test.sub_area == "Timirjazevskoe") &
             ((df_test.kitch_sq == 0))&
             (df_test.build_year < 10), 'build_year'] = round(df_train[(df_train.sub_area == "Timirjazevskoe")&
             (df_train.build_year > 10)].build_year.mean(),0)

df_train.loc[df_train['full_sq'] < 20, 'full_sq'] = np.NaN
df_test.loc[df_test['full_sq'] < 20, 'full_sq'] = np.NaN

df_train.loc[df_train['full_sq'] >= 1000, 'full_sq'] = np.NaN

df_train = df_train[(df_train.full_sq.notna()) & (df_train.num_room.notna())& (df_train.kitch_sq.notna())]

df_train.reset_index(drop=True, inplace=True)

df_train.loc[df_train['life_sq'] <= 1, 'life_sq'] = np.NaN
df_test.loc[df_test['life_sq'] <= 1, 'life_sq'] = np.NaN
df_train.loc[df_train['life_sq'].isna(), 'life_sq'] = df_train.full_sq -df_train.kitch_sq
df_test.loc[df_test['life_sq'].isna(), 'life_sq'] = df_test.full_sq -df_test.kitch_sq
df_test.loc[df_test['kitch_sq'].isna(), 'kitch_sq'] = df_test.full_sq -df_test.life_sq
df_test.loc[df_test['full_sq'].isna(), 'full_sq'] = df_test.kitch_sq +df_test.life_sq

df_train['ls_incorr'] = df_train.apply(lambda x: 1 if ((x.life_sq > x.full_sq) | (x.life_sq == 0)) else 0, axis=1)
df_train['life_sq'] = df_train.apply(lambda x: x.full_sq  - x.kitch_sq if x.ls_incorr else x.life_sq, axis=1)

df_test['ls_incorr'] = df_test.apply(lambda x: 1 if ((x.life_sq > x.full_sq) | (x.life_sq == 0)) else 0, axis=1)
df_test['life_sq'] = df_test.apply(lambda x: x.full_sq  - x.kitch_sq if x.ls_incorr else x.life_sq, axis=1)

df_train.drop(['ls_incorr'], axis=1, inplace=True)
df_test.drop(['ls_incorr'], axis=1, inplace=True)

df_train['ks_incorr'] = df_train.apply(lambda x: 1 if x.kitch_sq > x.full_sq else 0, axis=1)
df_train['kitch_sq'] = df_train.apply(lambda x: x.full_sq - x.life_sq if x.ks_incorr else x.kitch_sq, axis=1)

df_test['ks_incorr'] = df_test.apply(lambda x: 1 if x.kitch_sq > x.full_sq else 0, axis=1)
df_test['kitch_sq'] = df_test.apply(lambda x: x.full_sq - x.life_sq if x.ks_incorr else x.kitch_sq, axis=1)

df_train.drop(['ks_incorr'], axis=1, inplace=True)
df_test.drop(['ks_incorr'], axis=1, inplace=True)

df_train.loc[df_train.floor == 0, 'floor'] = 1
df_test.loc[df_test.floor == 0, 'floor'] = 1

df_train.loc[df_train.max_floor == 0, 'max_floor'] = df_train.floor
df_test.loc[df_test.max_floor == 0, 'max_floor'] = df_train.floor

df_train.loc[df_train.max_floor < df_train.floor, 'floor'] = df_train.max_floor
df_test.loc[df_test.max_floor < df_test.floor, 'floor'] = df_test.max_floor

df_test.floor = df_test.floor.fillna(round(df_train.groupby(['sub_area'])['floor'].transform('mean'), 0))

df_test.loc[df_test.max_floor.isna(), 'max_floor'] = df_test.floor

df_train.loc[df_train.state > 4, 'state'] = np.NaN
df_test.loc[df_test.state > 4, 'state'] = np.NaN

df_train.loc[df_train.index == 9322, 'num_room'] = round(df_train.life_sq/10,0)
df_train.loc[df_train.index == 82, 'num_room'] = round(df_train.life_sq/10,0)
df_train.loc[df_train.index == 8309, 'num_room'] = round(df_train.life_sq/10,0)
df_train.loc[df_train.index == 8590, 'num_room'] = round(df_train.life_sq/10,0)
df_train.loc[df_train.index == 4924, 'num_room'] = 4
df_train.loc[df_train.index == 8678, 'num_room'] = 1
df_train.loc[df_train.index == 9640, 'num_room'] = 6
df_train.loc[df_train.index == 11006, 'num_room'] = 1
df_train.loc[df_train.index == 11057, 'num_room'] = 2
df_train.loc[df_train.index == 13316, 'num_room'] = 4
df_test.loc[df_test.index == 61, 'num_room'] = 3
df_test.loc[df_test.index == 2585, 'num_room'] = 2
df_test.loc[df_test.index == 2873, 'num_room'] = 2

gr_df_t = df_train.groupby(['sub_area'])['full_all'].nunique().reset_index().sort_values('full_all')

gr_df_test = df_test.groupby(['sub_area'])['full_all'].nunique().reset_index().sort_values('full_all')

df_train = df_train.merge(df_macro, on='timestamp')[['id', 'timestamp', 'full_sq',
                                          'life_sq', 'kitch_sq', 'num_room',
                                          'floor', 'max_floor', 'state',
                                          'material', 'build_year', 'full_all',
                                          'sub_area', 'salary', 'fixed_basket',
                                          'rent_price_1room_eco',
                                          'rent_price_2room_eco',
                                          'rent_price_3room_eco',
                                          'average_life_exp', 'price_doc']]
df_test = df_test.merge(df_macro, on='timestamp', how='left')[['id', 'timestamp', 'full_sq',
                                          'life_sq', 'kitch_sq', 'num_room',
                                          'floor', 'max_floor', 'state',
                                          'material', 'build_year', 'full_all',
                                          'sub_area', 'salary', 'fixed_basket',
                                          'rent_price_1room_eco',
                                          'rent_price_2room_eco',
                                          'rent_price_3room_eco',
                                          'average_life_exp']]
os.mkdir("data/baselines/stage2")

df_train.to_csv('data/baselines/stage2/train.csv', index=False)
df_test.to_csv('data/baselines/stage2/test.csv', index=False)
df_macro.to_csv('data/baselines/stage2/macro.csv', index=False)