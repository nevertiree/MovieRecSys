# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
print(os.listdir("../data"))

data_path = os.path.abspath("../data")

# print(os.path.join(data_path,"movies.csv"))
df_movies = pd.read_csv("../data/movies.csv")
df_ratings = pd.read_csv("../data/ratings.csv")
df_tags = pd.read_csv("../data/tags.csv")
df_links = pd.read_csv("../data/links.csv")
# 合并imdbld，用于索引图片
df_ratings = df_ratings.merge(df_links, on='movieId')

# To Do

#合并movie
df_ratings = df_ratings.merge(df_movies, on='movieId')
df_ratings.head()

user_preference_topk = df_ratings.sort_values(
    by='rating', ascending=False).groupby(
        'userId')['movieId'].agg(lambda df: [id for id in df[0:3]])
df_ratings['user_preference_topk'] = df_ratings['userId'].map(
    user_preference_topk)

df_ratings.to_csv('data.csv')
