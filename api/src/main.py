from fastapi import FastAPI, HTTPException,Path,Query
from pydantic import BaseModel,Field
import numpy as np
import pandas as pd
import joblib
from api.src.utils_backend import get_items_rated_by_user
from pathlib import Path as FilePath
from api.src.recommender import ContentBasedFiltering,HybridRecommender,NBCF

current_file = FilePath(__file__).resolve()

project_root=current_file.parent.parent.parent

app=FastAPI(
    title='Movie Recommendation API',
    description='Microservice đề xuất phim',
    version='1.0.0'
)

mapping = pd.read_csv(f'{project_root}/data/processed/movie_mapping.csv')
rate_matrix = pd.read_csv(f'{project_root}/data/processed/ratings_a_train.csv').values
ratings_df=pd.read_csv(f'{project_root}/data/processed/ratings_a_train.csv')
cb_model = joblib.load(f'{project_root}/api/models/cb_model.pkl')
cf_model = joblib.load(f'{project_root}/api/models/cf_model.pkl')
hybrid_system = joblib.load(f'{project_root}/api/models/hb_model.pkl')

@app.get('/users/{user_id}/ratings')
async def get_items_by_user(user_id:int=Path(...,ge=0,le=942)):
    item_id,scores=get_items_rated_by_user(rate_matrix,user_id)
    
    return {
        'item_id':item_id.astype(int).tolist(),
        'scores':scores.astype(float).tolist()
    }

@app.get('/movies/{movie_id}/title')
async def get_movie_title_by_id(movie_id:int=Path(...,ge=0,le=1681)):
    return {
        'title':mapping.loc[movie_id,'movie title']
    }
    
@app.get('/movies/search')
async def search_all_movies_by_title(title:str=Query(...,min_length=1,max_length=50)):
    matches=mapping[mapping['movie title'].str.contains(title,case=False,na=False)]
    if not matches.empty:
        return {
            'movies': dict(zip(matches['movie id'].astype(int),matches['movie title']))
        }
    else:
        return {
            'movies':{}
        }

@app.get('/movies/top-rated')
async def get_top_rated_movies(min_ratings:int=50,top:int=10):
    stats=ratings_df.groupby('movie_id').agg(
        rating_mean=('rating','mean'),
        rating_count=('rating','count')
    ).reset_index()
    top_items=stats[stats['rating_count']>=min_ratings]
    top_items=top_items.sort_values('rating_mean',ascending=False).head(top)
    top_items['movie_id'] = top_items['movie_id'].astype(int)
    top_items['rating_count'] = top_items['rating_count'].astype(int)
    top_items['rating_mean'] = top_items['rating_mean'].astype(float)
    return {
        'top_items':top_items.to_dict(orient='records')
    }
    
@app.get('/movies/{movie_id}/ratings')
async def get_ratings_of_a_movie(movie_id:int=Path(...,ge=0,le=1681)):
    return {
        'ratings':ratings_df[ratings_df['movie_id']==movie_id]['rating'].astype(float).tolist()
    }

@app.get('/movies/{movie_id}/similar')
async def get_similar_movies(movie_id:int=Path(...,ge=0,le=1681),top:int=10,
                                           model_type:str='cb'):
    if model_type=='cb':
        recommendations=cb_model.recommend_similar_items(movie_id,top)
    else:
        recommendations=cf_model.recommend_similar_items(movie_id,top)
    return {
        'recommendations':recommendations.tolist()
    }

@app.get('/users/{user_id}/recommend')
async def get_recommend_movies_for_user(user_id:int=Path(...,ge=0,le=942),top:int=10):
    return {
        'movies':hybrid_system.recommend_for_user(user_id,top).tolist()
    }