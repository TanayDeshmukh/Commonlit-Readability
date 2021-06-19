import os
import pandas as pd

data_file = os.path.join('./data', 'train.csv')
df = pd.read_csv(data_file)
# print(df['target'][:10])
# print('--  ', df.loc[0, 'target'])
for i, row in df.iterrows():
    print(row['id'], row['target'], row['standard_error'])
    exit(0)
    