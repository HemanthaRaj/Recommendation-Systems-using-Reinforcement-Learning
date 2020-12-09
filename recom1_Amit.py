# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('C:/Users/Admin/Downloads/archive/movies_metadata.csv', low_memory = False)
display(data.head(4))

# Weighted rating(WR) = v/(v+m) * R + m/(v+m) * C
# v is the number of votes for the movie
# m is the minimum votes for it to be on the chart (80th percentile)
# C is the mean vote across the report
# R is the average rating of the movie

C = data['vote_average'].mean()
print(C)

m = data['vote_count'].quantile(0.8)
print(m)

movies_subset = data.copy().loc[data['vote_count'] >= m]
print(movies_subset.shape)

def weight_rating (y, m = m, C = C):
    v = y['vote_count']
    R = y['vote_average']
    z = v/(v+m) * R + m/(v+m) * C
    return z

movies_subset['score'] = movies_subset.apply(weight_rating, axis = 1)
movies_subset = movies_subset.sort_values('score', ascending=False)
display(movies_subset[['title','vote_count','vote_average','score']].head(10))

#import TFIDF module fom the scikit- learn
#eliminate stop words 
#replace NaNs
#construct the TFIDF matrix on the data

tfidf = TfidfVectorizer(stop_words = 'english')
data['overview'] = data['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)
print(tfidf.get_feature_names()[4000:4010])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix[:100], tfidf_matrix[:100])
print(cosine_sim)

indices = pd.Series(data.index, index = data['title']).drop_duplicates()
print(indices[:5])

def get_recommendation(title, cosine_sim = cosine_sim):
    index = pd.Series(data.index, index = data['title']).drop_duplicates()
    #print(index)
    sim_score = list(enumerate(cosine_sim[index]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:6]
    movie_indices = [j[0] for j in sim_score]
    return data['title'].iloc[movie_indices]

#get_recommendation('Toy Story')
    
    
