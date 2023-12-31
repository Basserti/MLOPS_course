#!/usr/bin/python3

import os
import pandas as pd


df_train = pd.read_csv('data/backup_data_not_used/train.csv')
df_test = pd.read_csv('data/backup_data_not_used/test.csv')
df_macro = pd.read_csv('data/backup_data_not_used/macro.csv')
submission = pd.read_csv('data/backup_data_not_used/submission.csv')

os.mkdir("data/raw")

df_train.to_csv('data/raw/train.csv', index=False)
df_test.to_csv('data/raw/test.csv', index=False)
df_macro.to_csv('data/raw/macro.csv', index=False)
submission.to_csv('data/raw/submission.csv', index=False)
