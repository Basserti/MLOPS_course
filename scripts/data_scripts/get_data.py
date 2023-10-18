#!/usr/bin/python3

import pandas as pd

df_train = pd.read_csv('/home/user1/project_mlops/MLOPS_course/data/backup_data_not_used/train.csv')
df_test = pd.read_csv('/home/user1/project_mlops/MLOPS_course/data/backup_data_not_used/test.csv')
df_macro = pd.read_csv('/home/user1/project_mlops/MLOPS_course/data/backup_data_not_used/macro.csv')
submission = pd.read_csv('/home/user1/project_mlops/MLOPS_course/data/backup_data_not_used/submission.csv')

df_train.to_csv('/home/user1/project_mlops/MLOPS_course/data/raw/train.csv', index=False)
df_test.to_csv('/home/user1/project_mlops/MLOPS_course/data/raw/test.csv', index=False)
df_macro.to_csv('/home/user1/project_mlops/MLOPS_course/data/raw/macro.csv', index=False)
submission.to_csv('/home/user1/project_mlops/MLOPS_course/data/raw/submission.csv', index=False)