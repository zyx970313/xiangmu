# encoding=utf-8

import pandas as pd

df = pd.read_csv('./movie_title.csv')
# print(df)

title = df['movie_title'].values
# print(type(title),title)
print(title[0])
print(title[1])


with open('./movie_title.txt', 'a') as f:
    for t in title[1:]:
        f.write(t + ' ' + 'nz' + '\n')
