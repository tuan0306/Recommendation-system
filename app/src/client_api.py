import requests
import pandas as pd
import numpy as np
 
BASE_URL='http://localhost:8000'

def get_top_rated_movies(min_ratings=50,top=10):
    API_URL=f'{BASE_URL}/movies/top-rated'
    response=requests.get(API_URL,params={'min_ratings':min_ratings,'top':top})
    response.raise_for_status()
    result=response.json()
    return pd.DataFrame(result['top_items'])

def search_all_movies_by_title(movie_query):
    API_URL=f'{BASE_URL}/movies/search'
    response=requests.get(API_URL,params={'title':movie_query})
    response.raise_for_status()
    result=response.json()
    movies_dict = result['movies']
    if not movies_dict:
        return pd.DataFrame()
    return pd.DataFrame(list(movies_dict.items()), columns=['movie_id', 'title'])

def get_movie_title_by_id(movie_id):
    API_URL=f'{BASE_URL}/movies/{movie_id}/title'
    response=requests.get(API_URL)
    response.raise_for_status()
    result=response.json()
    return result['title']

def get_ratings_of_a_movie(movie_id):
    API_URL=f'{BASE_URL}/movies/{movie_id}/ratings'
    response=requests.get(API_URL)
    response.raise_for_status()
    result=response.json()
    return np.array(result['ratings'])

def recommend_similar_items(movie_id,top=10,model_type='cb'):
    API_URL=f'{BASE_URL}/movies/{movie_id}/similar'
    response=requests.get(API_URL,params={'top':top,'model_type':model_type})
    response.raise_for_status()
    result=response.json()
    return np.array(result['recommendations'])

def get_items_rated_by_user(user_id):
    API_URL=f'{BASE_URL}/users/{user_id}/ratings'
    response=requests.get(API_URL)
    response.raise_for_status()
    result=response.json()
    return (np.array(result['item_id']),np.array(result['scores']))

def recommend_for_user(user_id,top=10):
    API_URL=f'{BASE_URL}/users/{user_id}/recommend'
    response=requests.get(API_URL,params={'top':top})
    response.raise_for_status()
    result=response.json()
    return np.array(result['movies'])